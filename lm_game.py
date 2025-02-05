import logging
import time
import dotenv
import os

# Suppress Gemini/PaLM gRPC warnings
os.environ['GRPC_PYTHON_LOG_LEVEL'] = '40'  # ERROR level only
import google.generativeai as genai  # Import after setting log level

from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format

# For concurrency:
import concurrent.futures

from lm_service_versus import load_model_client, assign_models_to_powers

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S"
)

def gather_possible_orders(game, power_name):
    """
    Returns a dictionary mapping each orderable location to the list of valid orders.
    """
    orderable_locs = game.get_orderable_locations(power_name)
    all_possible = game.get_all_possible_orders()

    result = {}
    for loc in orderable_locs:
        result[loc] = all_possible.get(loc, [])
    return result

def main():
    logger.info("Starting a new Diplomacy game for testing with multiple LLMs, now concurrent!")
    start_whole = time.time()

    # Create a standard Diplomacy game object
    game = Game()

    # Use assign_models_to_powers to get each power's LLM model_id
    power_model_map = assign_models_to_powers()

    # We'll let the game run until it's done OR we hit year 1905 for early stop
    max_year = 1905

    while not game.is_game_done:
        phase_start = time.time()
        current_phase = game.get_current_phase()  # e.g. "S1901M"
        logger.info(f"PHASE: {current_phase} (time so far: {phase_start - start_whole:.2f}s)")

        year_str = current_phase[1:5]
        year_int = int(year_str)
        if year_int > max_year:
            logger.info(f"Reached year {year_int}, stopping the test game early.")
            break

        active_powers = [
            (p_name, p_obj) for p_name, p_obj in game.powers.items()
            if not p_obj.is_eliminated()
        ]

        # Prepare concurrency
        futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_powers)) as executor:
            for power_name, power_obj in active_powers:
                model_id = power_model_map.get(power_name, "o3-mini")

                # Load the model client for this power
                client = load_model_client(model_id)

                # Gather possible orders
                possible_orders = gather_possible_orders(game, power_name)
                if not possible_orders:
                    logger.info(f"No orderable locations for {power_name}; skipping.")
                    continue

                board_state = game.get_state()

                # Submit a concurrent task to get_orders
                future = executor.submit(client.get_orders, board_state, power_name, possible_orders)
                futures[future] = power_name
                logger.debug(f"Submitted get_orders task for power {power_name} using {model_id}.")

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                p_name = futures[future]
                try:
                    orders = future.result()
                    logger.debug(f"Orders for {p_name}: {orders}")
                    if orders:
                        game.set_orders(p_name, orders)
                except Exception as exc:
                    logger.error(f"LLM request failed for {p_name}: {exc}")
                    # If there's an error, fallback or skip
                    # Or do nothing - the game can continue anyway

        logger.info("Processing orders...\n")
        game.process()
        logger.info("Phase complete.\n")

    # Once finished or forced to stop, save the final game state
    duration = time.time() - start_whole
    logger.info(f"Game ended after {duration:.2f}s. Saving to 'lmvsgame.json'.")
    to_saved_game_format(game, output_path='lmvsgame.json')

if __name__ == "__main__":
    main()
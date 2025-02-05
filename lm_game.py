import logging
import time
import dotenv
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format
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
    logger.info("Starting a new Diplomacy game for testing with multiple LLMs.")
    start_whole = time.time()

    # Create a standard Diplomacy game object
    game = Game()

    # No 'base model' needed here. We let each power pick its own model from our assignment dict.
    power_model_map = assign_models_to_powers()

    # We'll let the game run until it's done OR we hit the year 1905 (5 full seasons).
    max_year = 1905

    while not game.is_game_done:
        phase_start = time.time()
        current_phase = game.get_current_phase()  # e.g. "S1901M"
        logger.info(f"PHASE: {current_phase} (time so far: {phase_start - start_whole:.2f}s)")

        # Parse the year from the phase string, typically "S1901M" => year = 1901
        year_str = current_phase[1:5]
        year_int = int(year_str)

        # Stop early if we've exceeded our target year
        if year_int > max_year:
            logger.info(f"Reached year {year_int}, stopping the test game early.")
            break

        # Collect all active, non-eliminated powers
        active_powers = [
            (p_name, p_obj) for p_name, p_obj in game.powers.items()
            if not p_obj.is_eliminated()
        ]

        # For each active power, get orders using the assigned model
        for power_name, power_obj in active_powers:
            model_id = power_model_map.get(power_name)
            if not model_id:
                # If a power isn't in our dict, skip or set a fallback:
                logger.warning(f"No model assigned for {power_name}, using fallback 'o3-mini'.")
                model_id = "o3-mini"

            # Load the model client for this power
            client = load_model_client(model_id)

            # Gather order options
            possible_orders = gather_possible_orders(game, power_name)
            if not possible_orders:
                logger.info(f"No orderable locations for {power_name}; skipping.")
                continue

            board_state = game.get_state()
            orders = client.get_orders(board_state, power_name, possible_orders)
            game.set_orders(power_name, orders)

        logger.info("Processing orders...\n")
        game.process()
        logger.info("Phase complete.\n")

    # Once finished or forced to stop, save the final game state
    logger.info(f"Game ended after {time.time() - start_whole:.2f}s. Saving to 'lmvsgame.json'.")
    to_saved_game_format(game, output_path='lmvsgame.json')

if __name__ == "__main__":
    main()
import logging
import datetime    # we might also use datetime for manual timing
import dotenv
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format
from lm_service import LMService
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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
      {
        'PAR': ['A PAR H','A PAR - BUR','A PAR - GAS', ...],
        'BRE': [...],
        ...
      }
    """
    orderable_locs = game.get_orderable_locations(power_name)
    all_possible = game.get_all_possible_orders()

    result = {}
    for loc in orderable_locs:
        result[loc] = all_possible[loc] if loc in all_possible else []
    return result

def main():
    logger.info("Starting a new Diplomacy game using the LMService for orders.")
    start_whole = time.time()  # Track total time

    game = Game()  # Or specify a map_name=... if desired
    lm_service = LMService(model_name="o3-mini")

    while not game.is_game_done:
        phase_start = time.time()
        logger.info(f"PHASE: {game.get_current_phase()} (time so far: {phase_start - start_whole:.2f}s)")

        # Collect all active (non-eliminated) powers
        active_powers = [
            (power_name, p) for power_name, p in game.powers.items() 
            if not p.is_eliminated()
        ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_power = {}
            for power_name, power in active_powers:
                possible_orders = gather_possible_orders(game, power_name)
                if not possible_orders:
                    logger.info(f"No orderable locations for {power_name} this phase; skipping.")
                    continue

                board_state = game.get_state()

                # Log a checkpoint
                t_submit = time.time()
                logger.info(f"Submitting LLM job for {power_name} at {t_submit - start_whole:.2f}s")

                # Submit the LLM request
                future = executor.submit(
                    lm_service.get_orders,
                    board_state,
                    power_name,
                    possible_orders
                )
                future_to_power[future] = power_name

            for future in as_completed(future_to_power):
                p_name = future_to_power[future]
                t_return = time.time() - start_whole
                try:
                    orders = future.result()
                    logger.info(f"LLM orders for {p_name} arrived at {t_return:.2f}s: {orders}")
                    game.set_orders(p_name, orders)
                except Exception as exc:
                    logger.error(f"Error getting orders for {p_name}: {exc}")

        logger.info(f"Done collecting orders from all powers. Elapsed: {time.time() - start_whole:.2f}s")
        logger.info("Processing orders...\n")
        before_process = time.time() - start_whole
        game.process()
        after_process = time.time() - start_whole
        logger.info(f"Done with game.process() for this phase. That took {after_process - before_process:.2f}s (total {after_process:.2f}s)")

    logger.info(f"Game completed after {time.time() - start_whole:.2f}s. Saving final state to 'lm_game.json'.")
    to_saved_game_format(game, output_path='lm_game.json')

if __name__ == "__main__":
    main()
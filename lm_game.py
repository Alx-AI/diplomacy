import logging
import time
import dotenv
import os

# Suppress Gemini/PaLM gRPC warnings
os.environ['GRPC_PYTHON_LOG_LEVEL'] = '40'  # ERROR level only
import google.generativeai as genai  # Import after setting log level

from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format

# Added import: we'll create and add standard Diplomacy messages
from diplomacy.engine.message import Message, GLOBAL

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

def conduct_negotiations(game, max_rounds=3):
    """
    Conducts a round-robin conversation among all non-eliminated powers.
    Each power can send up to 'max_rounds' messages, one at a time.
    Instead of storing them in 'conversations', we create a
    diplomacy.engine.message.Message for each new LLM response and add it
    to game.messages (which will later be archived to message_history).
    """
    logger.info("Starting negotiation phase.")

    # Conversation messages are kept in a local list ONLY to build conversation_so_far text.
    # We no longer store them in 'game.conversations'.
    conversation_messages = []

    active_powers = [
        p_name for p_name, p_obj in game.powers.items() if not p_obj.is_eliminated()
    ]

    # We do up to 'max_rounds' single-message turns for each power
    for round_index in range(max_rounds):
        for power_name in active_powers:
            # Build the conversation context from all messages so far
            conversation_so_far = "\n".join(
                f"{msg['sender']}: {msg['content']}" for msg in conversation_messages
            )

            # Ask the LLM for a single reply
            client = load_model_client(game.power_model_map.get(power_name, "o3-mini"))
            new_message = client.get_conversation_reply(
                power_name=power_name,
                conversation_so_far=conversation_so_far,
                game_phase=game.current_short_phase
            )

            if new_message:
                # We log for debugging
                logger.info(f"Power {power_name} says:\n{new_message}")
                # Keep local record only for building future conversation context
                conversation_messages.append({
                    "sender": power_name,
                    "content": new_message.strip()
                })

                # Create an official public (global) message in the Diplomacy engine
                diplo_message = Message(
                    phase=game.current_short_phase,
                    sender=power_name,
                    recipient=GLOBAL,           # Everyone sees it
                    message=new_message.strip() # The LLM's content
                )
                game.add_message(diplo_message)

def main():
    logger.info("Starting a new Diplomacy game for testing with multiple LLMs, now concurrent!")
    start_whole = time.time()

    # Create a fresh Diplomacy game
    game = Game()

    # Map each power to its chosen LLM
    game.power_model_map = assign_models_to_powers()

    max_year = 1902

    while not game.is_game_done:
        phase_start = time.time()
        current_phase = game.get_current_phase()
        logger.info(f"PHASE: {current_phase} (time so far: {phase_start - start_whole:.2f}s)")

        # DEBUG: Print the short phase to confirm
        logger.info(f"DEBUG: current_short_phase is '{game.current_short_phase}'")

        # Prevent unbounded sim
        year_str = current_phase[1:5]
        year_int = int(year_str)
        if year_int > max_year:
            logger.info(f"Reached year {year_int}, stopping the test game early.")
            break

        # Use endswith("M") for movement phases (like F1901M, S1902M)
        if game.current_short_phase.endswith("M"):
            logger.info("Starting negotiation phase block...")
            conduct_negotiations(game, max_rounds=3)

        # Gather orders from each power concurrently
        active_powers = [
            (p_name, p_obj) for p_name, p_obj in game.powers.items()
            if not p_obj.is_eliminated()
        ]

        # Then proceed with concurrent order generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_powers)) as executor:
            futures = {}
            for power_name, _ in active_powers:
                model_id = game.power_model_map.get(power_name, "o3-mini")
                client = load_model_client(model_id)
                possible_orders = gather_possible_orders(game, power_name)
                if not possible_orders:
                    logger.info(f"No orderable locations for {power_name}; skipping.")
                    continue
                board_state = game.get_state()
                future = executor.submit(
                    client.get_orders, board_state, power_name, possible_orders
                )
                futures[future] = power_name
                logger.debug(f"Submitted get_orders task for power {power_name}.")

            for future in concurrent.futures.as_completed(futures):
                p_name = futures[future]
                try:
                    orders = future.result()
                    logger.debug(f"Orders for {p_name}: {orders}")
                    if orders:
                        game.set_orders(p_name, orders)
                except Exception as exc:
                    logger.error(f"LLM request failed for {p_name}: {exc}")

        logger.info("Processing orders...\n")
        game.process()
        logger.info("Phase complete.\n")

    duration = time.time() - start_whole
    logger.info(f"Game ended after {duration:.2f}s. Saving to 'lmvsgame.json'.")
    # Save the game to a JSON file
    output_path = 'lmvsgame.json'
    if not os.path.exists(output_path):
        to_saved_game_format(game, output_path=output_path)
    else:
        logger.info("Game file already exists, saving with unique filename.")
        output_path = f'{output_path}_{time.strftime("%Y%m%d_%H%M%S")}.json'
        to_saved_game_format(game, output_path=output_path)
if __name__ == "__main__":
    main()
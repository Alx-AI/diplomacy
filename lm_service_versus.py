import os
import openai
import logging
import dotenv
import json
import re
dotenv.load_dotenv()

logger = logging.getLogger(__name__)
client = openai.OpenAI()

class LMService:
    """
    A minimal class that queries an OpenAI model (such as 'o3-mini') to produce Diplomacy orders.
    Uses few-shot and chain-of-thought in the prompt. Falls back if parsing fails.
    """

    def __init__(self, model_name="o3-mini"):
        self.model_name = model_name
        # In a real environment, ensure openai.api_key is set, e.g.:
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def build_prompt(self, board_state, power_name, possible_orders):
        """
        Builds a prompt string for the LLM. We provide context on the board state,
        plus instructions to produce a PARSABLE OUTPUT...
        """

        # Summarize relevant pieces of board_state for better strategic context:
        # You can refine these details more or less, depending on what you want the LLM to see.
        # For example, list your units, centers, adjacent powers, etc.
        # We'll keep it short for demonstration.
        units_info = board_state["units"].get(power_name, [])
        centers_info = board_state["centers"].get(power_name, [])
        year_phase = board_state["phase"]  # e.g. 'S1901M'

        # Construct a short summary:
        summary = (
            f"Power: {power_name}\n"
            f"Current Phase: {year_phase}\n"
            f"Your Units: {units_info}\n"
            f"Your Centers: {centers_info}\n"
            f"Possible Orders:\n"
        )
        for loc, orders in possible_orders.items():
            summary += f"  {loc}: {orders}\n"

        # Provide an example or chain-of-thought snippet,
        # then mandate final output must have PARSABLE OUTPUT JSON:
        few_shot_example = """
--- EXAMPLE ---
Power: FRANCE
Phase: S1901M
Your Units: ['A PAR','F BRE']
Possible Orders:
  PAR: ['A PAR H','A PAR - BUR','A PAR - GAS']
  BRE: ['F BRE H','F BRE - MAO']

Chain-of-thought: 
[insert detailed chain-of-thought here with your goal being to win given the information you have. 
Change your strategy if you are losing or winning. 
Depending on the situation, you may want to attack or defend. 
You may want to support your allies or attack your enemies.
The phase of the game tells you what year it is and what the current season is.
This is important because the rules of diplomacy change at the start of each season.
Channel best strategy for the current phase of the game.
]

PARSABLE OUTPUT:{{
  "orders": ["A PAR - BUR","F BRE - MAO"]
}}
--- END EXAMPLE ---
"""

        # The final instructions to the model
        instructions = (
            "IMPORTANT:\n"
            "Return EXACTLY one JSON block in the format:\n"
            "PARSABLE OUTPUT:{{\n"
            '  "orders": ["..."]\n'
            "}}\n"
            "Do not include explanation outside that block.\n"
        )

        # Combine everything
        prompt = (
            summary
            + few_shot_example
            + "\nNow produce the final orders for this power.\n"
            + instructions
        )
        return prompt

    def get_orders(self, board_state, power_name, possible_orders):
        prompt = self.build_prompt(board_state, power_name, possible_orders)
        logger.debug(f"Prompt for {power_name}:\n{prompt}")

        try:
            # Example usage - you may have different openai usage or a custom client
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()
            logger.info(f"Raw LLM response for {power_name}:\n{content}")

            # --- 1) Attempt direct JSON parse (like before). ---
            # We look for PARSABLE OUTPUT:{{ ...
            block_match = re.search(r'PARSABLE\s*OUTPUT:\s*\{\{\s*(.*?)\s*\}\}', content, flags=re.DOTALL)
            if block_match:
                snippet = block_match.group(1)
                snippet = snippet.replace("‘", '"').replace("’", '"')
                snippet = snippet.replace("“", '"').replace("”", '"')
                try:
                    data = json.loads(snippet)
                    moves = data["orders"]
                    return self._validate_orders(moves, possible_orders)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Strict JSON parse failed for {power_name}: {e}")

            # --- 2) If the above fails, fallback to simpler bracket-locating approach. ---
            # We'll search for a substring that starts with [' or [" and ends with ].
            # Then we parse it as a python list of strings.
            bracket_match = re.search(r'(\[[^]]*\])', content)
            if bracket_match:
                possible_list_str = bracket_match.group(1)
                logger.info(f"Attempting bracket-based parse: {possible_list_str}")
                # Clean up quotes
                possible_list_str = possible_list_str.replace("‘", '"').replace("’", '"')
                possible_list_str = possible_list_str.replace("“", '"').replace("”", '"')

                try:
                    moves = eval(possible_list_str)  # Danger in real code, but for demonstration...
                    if isinstance(moves, list):
                        return self._validate_orders(moves, possible_orders)
                except Exception as e:
                    logger.warning(f"Bracket-based parse also failed for {power_name}: {e}")

            # --- 3) If all else fails, fallback. ---
            logger.warning(f"Unable to parse any move list for {power_name}. Using fallback.")
            return self.fallback_orders(possible_orders)

        except Exception as exc:
            logger.error(f"Error in LLM or parse for {power_name}: {exc}")
            return self.fallback_orders(possible_orders)

    def _validate_orders(self, moves, possible_orders):
        """
        Take a list of moves from the LLM output, filter out invalid ones, fill missing with HOLD.
        """
        logger.debug(f"Proposed LLM moves: {moves}")
        validated = []

        # Mark location codes we used
        used_locs = set()

        if not isinstance(moves, list):
            logger.debug("LLM moves was not a list. Fallback.")
            return self.fallback_orders(possible_orders)

        for move in moves:
            move_str = move.strip()
            # Check if it's exactly in possible orders
            if any(move_str in loc_orders for loc_orders in possible_orders.values()):
                validated.append(move_str)
                # Extract location from e.g. "A BUD H" => BUD
                parts = move_str.split()
                if len(parts) >= 2:
                    used_locs.add(parts[1][:3])  # e.g. BUD
            else:
                logger.debug(f"Invalid move from LLM: {move_str} not in possible_orders.")

        # For any location not covered, do hold if available
        for loc, orders_list in possible_orders.items():
            if loc not in used_locs and orders_list:
                hold_candidates = [o for o in orders_list if o.endswith("H")]
                validated.append(hold_candidates[0] if hold_candidates else orders_list[0])

        if not validated:
            logger.warning("All LLM moves invalid, returning fallback.")
            return self.fallback_orders(possible_orders)

        logger.debug(f"Validated orders: {validated}")
        return validated

    def fallback_orders(self, possible_orders):
        """
        Fallback if we can't parse:
          - For each location, pick 'H' if present, else first possible move
        """
        fallback = []
        for loc, orders_list in possible_orders.items():
            holds = [o for o in orders_list if o.endswith("H")]
            fallback.append(holds[0] if holds else orders_list[0]) if orders_list else None
        return fallback

    def _call_openai(self, prompt):
        """
        Example wrapper for openai library calls. Replace with actual usage or your custom client.
        """
        import openai
        # Could add temperature, top_p, etc. as needed
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        ) 
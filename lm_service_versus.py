import os
import json
import re
import logging

from typing import List, Dict
from dotenv import load_dotenv

# Anthropics
import anthropic

# Google Generative AI
import google.generativeai as genai

# DeepSeek
from openai import OpenAI as DeepSeekOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

##############################################################################
# 1) Base Interface
##############################################################################
class BaseModelClient:
    """
    Base interface for any LLM client we want to plug in.
    Each must provide:
      - generate_response(prompt: str) -> str
      - get_orders(board_state, power_name, possible_orders) -> List[str]
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        """
        Returns a raw string from the LLM.
        Subclasses override this.
        """
        raise NotImplementedError("Subclasses must implement generate_response().")

    def build_prompt(self, board_state, power_name: str, possible_orders: Dict[str, List[str]]) -> str:
        """
        Unified prompt approach with 'PARSABLE OUTPUT' requirements.
        """
        units_info = board_state["units"].get(power_name, [])
        centers_info = board_state["centers"].get(power_name, [])
        year_phase = board_state["phase"]  # e.g. 'S1901M'

        summary = (
            f"Power: {power_name}\n"
            f"Current Phase: {year_phase}\n"
            f"Your Units: {units_info}\n"
            f"Your Centers: {centers_info}\n"
            f"Possible Orders:\n"
        )
        for loc, orders in possible_orders.items():
            summary += f"  {loc}: {orders}\n"

        few_shot_example = """
--- EXAMPLE ---
Power: FRANCE
Phase: S1901M
Your Units: ['A PAR','F BRE']
Possible Orders:
  PAR: ['A PAR H','A PAR - BUR','A PAR - GAS']
  BRE: ['F BRE H','F BRE - MAO']

Chain-of-thought:
[Be consistent with your secret chain-of-thought here, but do not reveal it. 
Aim for best strategic moves based on the possible orders, 
and produce an output in PARSABLE JSON format as shown below.]

PARSABLE OUTPUT:{
  "orders": ["A PAR - BUR","F BRE - MAO"]
}
--- END EXAMPLE ---
"""

        instructions = (
            "IMPORTANT:\n"
            "Return your thoughts and how you came up with the orders before ending with EXACTLY one JSON block in the format:\n"
            "PARSABLE OUTPUT:{\n"
            '  "orders": ["..."]\n'
            "}\n"
            "Include your explanation outside that block but make SURE to include your orders in the matching JSON block.\n"
        )

        prompt = (
            summary
            + few_shot_example
            + "\nNow think through the existing situation and produce the final orders for this power.\n"
            + instructions
        )
        return prompt

    def get_orders(self, board_state, power_name, possible_orders) -> List[str]:
        """
        High-level method:
         1) build_prompt
         2) call generate_response
         3) parse the "PARSABLE OUTPUT" JSON
         4) validate moves
         5) fallback if needed
        """
        # Build prompt
        prompt = self.build_prompt(board_state, power_name, possible_orders)
        # Query model
        try:
            content = self.generate_response(prompt)
            logger.info(f"[{self.model_name}] Raw LLM response for {power_name}:\n{content}")

            # Attempt direct JSON parse
            block_match = re.search(
                r'PARSABLE\s*OUTPUT:\s*\{\s*(.*?)\s*\}',
                content, flags=re.DOTALL
            )
            if block_match:
                snippet = block_match.group(1)
                snippet = snippet.replace("'", '"').replace("'", '"')
                snippet = snippet.replace("\"", '"').replace("\"", '"')
                try:
                    data = json.loads("{" + snippet + "}")
                    moves = data["orders"]
                    return self._validate_orders(moves, possible_orders)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"{self.model_name} JSON parse failed for {power_name}: {e}")

            # If direct parse fails, fallback to bracket-based
            bracket_match = re.search(r'(\[[^]]*\])', content)
            if bracket_match:
                possible_list_str = bracket_match.group(1)
                possible_list_str = (
                    possible_list_str
                    .replace("'", '"')
                    .replace("'", '"')
                    .replace("\"", '"')
                    .replace("\"", '"')
                )
                try:
                    # Danger in production; safe for demonstration
                    moves = eval(possible_list_str)
                    if isinstance(moves, list):
                        return self._validate_orders(moves, possible_orders)
                except Exception as e:
                    logger.warning(f"{self.model_name} bracket parse failed for {power_name}: {e}")

            logger.warning(f"[{self.model_name}] Unable to parse response for {power_name}. Using fallback.")
            return self.fallback_orders(possible_orders)

        except Exception as exc:
            logger.error(f"[{self.model_name}] LLM error for {power_name}: {exc}")
            return self.fallback_orders(possible_orders)

    def _validate_orders(self, moves: List[str], possible_orders: Dict[str, List[str]]) -> List[str]:
        """
        Filter out invalid moves, fill missing with HOLD, else fallback.
        """
        logger.debug(f"[{self.model_name}] Proposed LLM moves: {moves}")
        validated = []
        used_locs = set()

        if not isinstance(moves, list):
            logger.debug(f"[{self.model_name}] Moves not a list, fallback.")
            return self.fallback_orders(possible_orders)

        for move in moves:
            move_str = move.strip()
            # Check if it's in possible orders
            if any(move_str in loc_orders for loc_orders in possible_orders.values()):
                validated.append(move_str)
                parts = move_str.split()
                if len(parts) >= 2:
                    used_locs.add(parts[1][:3])
            else:
                logger.debug(f"[{self.model_name}] Invalid move from LLM: {move_str}")

        # Fill missing with hold
        for loc, orders_list in possible_orders.items():
            if loc not in used_locs and orders_list:
                hold_candidates = [o for o in orders_list if o.endswith("H")]
                validated.append(hold_candidates[0] if hold_candidates else orders_list[0])

        if not validated:
            logger.warning(f"[{self.model_name}] All moves invalid, fallback.")
            return self.fallback_orders(possible_orders)

        logger.debug(f"[{self.model_name}] Validated moves: {validated}")
        return validated

    def fallback_orders(self, possible_orders: Dict[str, List[str]]) -> List[str]:
        """
        Just picks HOLD if possible, else first option.
        """
        fallback = []
        for loc, orders_list in possible_orders.items():
            if orders_list:
                holds = [o for o in orders_list if o.endswith("H")]
                fallback.append(holds[0] if holds else orders_list[0])
        return fallback


##############################################################################
# 2) Concrete Implementations
##############################################################################

class OpenAIClient(BaseModelClient):
    """
    For 'o3-mini', 'gpt-4o', or other OpenAI model calls.
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)
        from openai import OpenAI  # Import the new client
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate_response(self, prompt: str) -> str:
        # Updated to new API format
        system_prompt = """
        You are a Diplomacy expert.
        You are given a board state and a list of possible orders for a power.
        You need to produce the final orders for that power.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()


class ClaudeClient(BaseModelClient):
    """
    For 'claude-2', 'claude-3-5-sonnet-20241022', etc.
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    def generate_response(self, prompt: str) -> str:
        system_prompt = """
        You are a Diplomacy expert.
        You are given a board state and a list of possible orders for a power.
        You need to produce the final orders for that power.
        """
        # Updated Claude messages format
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2000,
            system=system_prompt,  # system is now a top-level parameter
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text if response.content else ""


class GeminiClient(BaseModelClient):
    """
    For 'gemini-1.5-flash' or other Google Generative AI models.
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        # Updated config without system_instruction
        self.generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 2000,
        }

    def generate_response(self, prompt: str) -> str:
        # Add system prompt to the actual prompt for Gemini
        system_prompt = """
        You are a Diplomacy expert.
        You are given a board state and a list of possible orders for a power.
        You need to produce the final orders for that power.

        """  # Extra newline for separation
        full_prompt = system_prompt + prompt
        
        model = genai.GenerativeModel(
            self.model_name,
            generation_config=self.generation_config
        )
        response = model.generate_content(full_prompt)
        return response.text.strip() if response and response.text else ""


class DeepSeekClient(BaseModelClient):
    """
    For a hypothetical 'deepseek-reasoner' model (via OpenAI-like interface).
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.client = DeepSeekOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/"
        )

    def generate_response(self, prompt: str) -> str:
        # Similar to ChatCompletion
        system_prompt = """
        You are a Diplomacy expert.
        You are given a board state and a list of possible orders for a power.
        You need to produce the final orders for that power.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        if not response or not response.choices:
            logger.warning("[DeepSeek] No valid response.")
            return ""
        return response.choices[0].message.content.strip()


##############################################################################
# 3) Factory to Load Model Client
##############################################################################

def load_model_client(model_id: str) -> BaseModelClient:
    """
    Returns the appropriate LLM client for a given model_id string.
    Example usage:
       client = load_model_client("claude-3-5-sonnet-20241022")
    """
    # Basic pattern matching or direct mapping
    lower_id = model_id.lower()
    if "claude" in lower_id:
        return ClaudeClient(model_id)
    elif "gemini" in lower_id:
        return GeminiClient(model_id)
    elif "deepseek" in lower_id:
        return DeepSeekClient(model_id)
    else:
        # Default to OpenAI
        return OpenAIClient(model_id)


##############################################################################
# 4) Example Usage in a Diplomacy "main" or Similar
##############################################################################

def assign_models_to_powers():
    """
    Example usage: define which model each power uses.
    Return a dict: { power_name: model_id, ... }
    POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
    """
    return {
        "FRANCE": "o3-mini",
        "GERMANY": "claude-3-5-sonnet-20241022",
        "ENGLAND": "gemini-1.5-flash",
        "RUSSIA": "deepseek-reasoner",
        "ITALY": "gpt-4o",
        "AUSTRIA": "gpt-4o-mini",
        "TURKEY": "claude-3-5-haiku-20241022",
    }

def example_game_loop(game):
    """
    Pseudocode: Integrate with the Diplomacy loop.
    """
    # Suppose we gather all active powers
    active_powers = [(p_name, p_obj) for p_name, p_obj in game.powers.items() if not p_obj.is_eliminated()]
    power_model_mapping = assign_models_to_powers()

    for power_name, power_obj in active_powers:
        model_id = power_model_mapping.get(power_name, "o3-mini")
        client = load_model_client(model_id)

        # Get possible orders from the game
        possible_orders = game.get_all_possible_orders()
        board_state = game.get_state()

        # Get orders from the client
        orders = client.get_orders(board_state, power_name, possible_orders)
        game.set_orders(power_name, orders)

    # Then process, etc.
    game.process()

class LMServiceVersus:
    """
    Optional wrapper class if you want extra control.
    For example, you could store or reuse clients, etc.
    """
    def __init__(self):
        self.power_model_map = assign_models_to_powers()
    
    def get_orders_for_power(self, game, power_name):
        model_id = self.power_model_map.get(power_name, "o3-mini")
        client = load_model_client(model_id)
        possible_orders = gather_possible_orders(game, power_name)
        board_state = game.get_state()
        return client.get_orders(board_state, power_name, possible_orders) 
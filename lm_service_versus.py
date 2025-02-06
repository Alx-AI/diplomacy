import os
import json
import re
import logging
import ast

from typing import List, Dict, Optional
from dotenv import load_dotenv

# Anthropics
import anthropic

# Google Generative AI
# Suppress Gemini/PaLM gRPC warnings
os.environ['GRPC_PYTHON_LOG_LEVEL'] = '40'  # ERROR level only
import google.generativeai as genai  # Import after setting log level

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

    def get_orders(self, board_state, power_name: str, possible_orders: Dict[str, List[str]]) -> List[str]:
        """
        1) Builds the prompt.
        2) Calls LLM (generate_response).
        3) Tries to parse the "PARSABLE OUTPUT: { ... }" JSON.
        4) Fills missing or invalid with fallback.
        """
        prompt = self.build_prompt(board_state, power_name, possible_orders)
        raw_response = ""

        try:
            raw_response = self.generate_response(prompt)
            logger.info(f"[{self.model_name}] Raw LLM response for {power_name}:\n{raw_response}")

            # Attempt to parse the final "orders" from the LLM
            move_list = self._extract_moves(raw_response, power_name)
            if not move_list:
                logger.warning(f"[{self.model_name}] Could not extract moves for {power_name}. Using fallback.")
                return self.fallback_orders(possible_orders)

            # Validate or fallback
            validated_moves = self._validate_orders(move_list, possible_orders)
            return validated_moves

        except Exception as e:
            logger.error(f"[{self.model_name}] LLM error for {power_name}: {e}")
            return self.fallback_orders(possible_orders)

    def _extract_moves(self, raw_response: str, power_name: str) -> Optional[List[str]]:
        """
        Attempt multiple parse strategies to find JSON array of moves.
        
        1. Regex for PARSABLE OUTPUT lines.
        2. If that fails, also look for fenced code blocks with { ... }.
        3. Attempt bracket-based fallback if needed.
        
        Returns a list of move strings or None if everything fails.
        """
        # 1) Regex for "PARSABLE OUTPUT:{...}"
        pattern = r"PARSABLE OUTPUT\s*:\s*\{(.*?)\}\s*$"
        matches = re.search(pattern, raw_response, re.DOTALL)
        if not matches:
            # Some LLMs might not put the colon or might have triple backtick fences.
            logger.debug(f"[{self.model_name}] Regex parse #1 failed for {power_name}. Trying alternative patterns.")
            
            # 1b) Check for inline JSON after "PARSABLE OUTPUT"
            pattern_alt = r"PARSABLE OUTPUT\s*\{(.*?)\}\s*$"
            matches = re.search(pattern_alt, raw_response, re.DOTALL)

        # 2) If still no match, check for triple-backtick code fences containing JSON
        if not matches:
            code_fence_pattern = r"```json\s*\{(.*?)\}\s*```"
            matches = re.search(code_fence_pattern, raw_response, re.DOTALL)
            if matches:
                logger.debug(f"[{self.model_name}] Found triple-backtick JSON block for {power_name}.")
        
        # 3) Attempt to parse JSON if we found anything
        json_text = None
        if matches:
            # Add braces back around the captured group
            json_text = "{%s}" % matches.group(1).strip()
            json_text = json_text.strip()

        if not json_text:
            logger.debug(f"[{self.model_name}] No JSON text found in LLM response for {power_name}.")
            return None

        # 3a) Try JSON loading
        try:
            data = json.loads(json_text)
            return data.get("orders", None)
        except json.JSONDecodeError as e:
            logger.warning(f"[{self.model_name}] JSON decode failed for {power_name}: {e}. Trying bracket fallback.")

        # 3b) Attempt bracket fallback: we look for the substring after "orders"
        #     E.g. "orders: ['A BUD H']" and parse it. This is risky but can help with minor JSON format errors.
        #     We only do this if we see something like "orders": ...
        bracket_pattern = r'["\']orders["\']\s*:\s*\[([^\]]*)\]'
        bracket_match = re.search(bracket_pattern, json_text, re.DOTALL)
        if bracket_match:
            try:
                raw_list_str = "[" + bracket_match.group(1).strip() + "]"
                moves = ast.literal_eval(raw_list_str)
                if isinstance(moves, list):
                    return moves
            except Exception as e2:
                logger.warning(f"[{self.model_name}] Bracket fallback parse also failed for {power_name}: {e2}")

        # If all attempts failed
        return None

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

    def get_conversation_reply(self, power_name: str, conversation_so_far: str, game_phase: str) -> str:
        """
        Produce a single message (either private or global) from the model,
        given existing conversation context.
        """
        prompt = f"""You are {power_name} in a Diplomacy game, currently in phase {game_phase}.
You can send either a global message (visible to all powers) or a private message (visible only to a specific power).

Previous messages:
{conversation_so_far}

Instructions:
1. Decide whether to send a global message or a private message to a specific power.
2. Your response must be in valid JSON format as shown in the examples.
3. For private messages, choose your recipient carefully based on strategic value.
4. Keep messages concise and diplomatic.

Example response formats:
1. For a global message:
{{
    "message_type": "global",
    "content": "I propose we all work together against Turkey."
}}

2. For a private message:
{{
    "message_type": "private",
    "recipient": "FRANCE",
    "content": "Let's form a secret alliance against Germany."
}}

Think strategically about your diplomatic position and respond with your message in the correct JSON format:"""
        return self.generate_response(prompt)

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
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
            )
            if not response or not hasattr(response, "choices") or not response.choices:
                logger.warning(f"[{self.model_name}] Empty or invalid result in generate_response. Returning empty.")
                return ""
            return response.choices[0].message.content.strip()
        except json.JSONDecodeError as json_err:
            logger.error(f"[{self.model_name}] JSON decoding failed in generate_response: {json_err}")
            return ""
        except Exception as e:
            logger.error(f"[{self.model_name}] Unexpected error in generate_response: {e}")
            return ""

    def get_conversation_reply(self, power_name: str, conversation_so_far: str, game_phase: str) -> str:
        """
        Produces a single message from the LLM in plain text form, with robust error handling.
        """
        import json
        from json.decoder import JSONDecodeError

        system_prompt = f"""
        You are playing as {power_name} in a Diplomacy negotiation during phase {game_phase}.
        You have read all messages so far. Now produce a single new message with your strategy or statement.
        IMPORTANT: Output only the text of your message, with no additional formatting or disclaimers.
        """
        conversation_prompt = f"Conversation so far:\n{conversation_so_far}\n\nYour new message:"

        try:
            # Perform the request
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation_prompt}
                ],
                max_completion_tokens=2000
            )

            # If there's no valid response or choices, return empty
            if not response or not hasattr(response, "choices") or not response.choices:
                logger.warning(f"[{self.model_name}] Empty or invalid response for {power_name}. Returning empty.")
                return ""

            # Attempt to parse the content (OpenAI library usually does this, but we add a safety net)
            return response.choices[0].message.content.strip()

        except JSONDecodeError as json_err:
            logger.error(f"[{self.model_name}] JSON decoding failed for {power_name}: {json_err}")
            return ""  # Fallback
        except Exception as e:
            logger.error(f"[{self.model_name}] Unexpected error for {power_name}: {e}")
            return ""

class ClaudeClient(BaseModelClient):
    """
    For 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', etc.
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
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2000,
                system=system_prompt,  # system is now a top-level parameter
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            if not response.content:
                logger.warning(f"[{self.model_name}] Empty content in Claude generate_response. Returning empty.")
                return ""
            return response.content[0].text.strip() if response.content else ""
        except json.JSONDecodeError as json_err:
            logger.error(f"[{self.model_name}] JSON decoding failed in generate_response: {json_err}")
            return ""
        except Exception as e:
            logger.error(f"[{self.model_name}] Unexpected error in generate_response: {e}")
            return ""

    def get_conversation_reply(self, power_name: str, conversation_so_far: str, game_phase: str) -> str:
        system_prompt = f"You are playing as {power_name} in this Diplomacy negotiation phase {game_phase}."
        user_prompt = (
            f"Conversation so far:\n{conversation_so_far}\n"
            "Please provide a single, short message in plain text. "
        )
        try:
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=2000
            )
            if not response.content:
                logger.warning(f"[{self.model_name}] No content in Claude conversation. Returning empty.")
                return ""
            return response.content[0].text.strip()
        except json.JSONDecodeError as json_err:
            logger.error(f"[{self.model_name}] JSON decoding failed in conversation: {json_err}")
            return ""
        except Exception as e:
            logger.error(f"[{self.model_name}] Unexpected error in conversation: {e}")
            return ""

class GeminiClient(BaseModelClient):
    """
    For 'gemini-1.5-flash' or other Google Generative AI models.
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self.generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 2000,
        }

    def generate_response(self, prompt: str) -> str:
        system_prompt = """
        You are a Diplomacy expert.
        You are given a board state and a list of possible orders for a power.
        You need to produce the final orders for that power.
        """
        full_prompt = system_prompt + prompt
        
        try:
            model = genai.GenerativeModel(
                self.model_name,
                generation_config=self.generation_config
            )
            response = model.generate_content(full_prompt)
            if not response or not response.text:
                logger.warning(f"[{self.model_name}] Empty Gemini generate_response. Returning empty.")
                return ""
            return response.text.strip()
        except Exception as e:
            logger.error(f"[{self.model_name}] Error in Gemini generate_response: {e}")
            return ""

    def get_conversation_reply(self, power_name: str, conversation_so_far: str, game_phase: str) -> str:
        """
        Produce a single short conversation message from the Gemini model, 
        given existing conversation context.
        """
        # Similar approach: create a system plus user prompt, then call model.generate_content
        system_prompt = f"You are playing as {power_name} in this Diplomacy negotiation phase {game_phase}.\n"
        user_prompt = (
            f"Conversation so far:\n{conversation_so_far}\n"
            "Please provide a single, short message in plain text. "
        )
        full_prompt = system_prompt + user_prompt

        try:
            model = genai.GenerativeModel(
                self.model_name,
                generation_config=self.generation_config
            )
            response = model.generate_content(full_prompt)
            if not response or not response.text:
                logger.warning(f"[{self.model_name}] Empty Gemini conversation response. Returning empty.")
                return ""
            return response.text.strip()
        except Exception as e:
            logger.error(f"[{self.model_name}] Error in Gemini get_conversation_reply: {e}")
            return ""

class DeepSeekClient(BaseModelClient):
    """
    For DeepSeek R1 'deepseek-reasoner'
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
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            if not response or not response.choices:
                logger.warning("[DeepSeek] No valid response in generate_response.")
                return ""
            return response.choices[0].message.content.strip()
        
        except json.JSONDecodeError as json_err:
            logger.error(f"[DeepSeek] JSON decode error in generate_response: {json_err}")
            return ""
        except Exception as e:
            logger.error(f"[DeepSeek] Unexpected error in generate_response: {e}")
            return ""
    
    def get_conversation_reply(self, power_name: str, conversation_so_far: str, game_phase: str) -> str:
        system_prompt = f"You are playing as {power_name} in this Diplomacy negotiation phase {game_phase}."
        user_prompt = (
            f"Conversation so far:\n{conversation_so_far}\n"
            "Please provide a single, short message in plain text. "
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": user_prompt}],
                max_completion_tokens=2000
            )
            if not response or not response.choices:
                logger.warning("[DeepSeek] No valid choices in conversation reply.")
                return ""
            return response.choices[0].message.content.strip()
        except json.JSONDecodeError as json_err:
            logger.error(f"[DeepSeek] JSON decode error in conversation: {json_err}")
            return ""
        except Exception as e:
            logger.error(f"[DeepSeek] Unexpected error in conversation: {e}")
            return ""  


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
    # "RUSSIA": "deepseek-reasoner",
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
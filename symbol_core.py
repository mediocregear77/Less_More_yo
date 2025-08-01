import json
import os
from datetime import datetime
from pathlib import Path
import asyncio

# --- Configuration Constants ---
# Paths to the knowledge base files. Using pathlib for OS-agnostic paths.
KB_BASE_DIR = Path("expression_core")
SYMBOL_KB_PATH = KB_BASE_DIR / "datasets" / "symbols" / "symbolic_meanings.json"
METAPHOR_MAP_PATH = KB_BASE_DIR / "datasets" / "metaphors" / "metaphorical_mappings.json"

class SymbolInterpreter:
    """
    Interprets symbolic meanings based on a knowledge base and contextual data.

    This class provides methods to load and manage symbolic and metaphorical
    knowledge, with a focus on dynamic, LLM-powered contextual interpretation.
    """
    def __init__(self):
        """Initializes the interpreter by loading the knowledge bases."""
        self.symbol_kb = self._load_json_file(SYMBOL_KB_PATH)
        self.metaphor_map = self._load_json_file(METAPHOR_MAP_PATH)
        # Placeholder for API key, which will be provided by the runtime
        self.api_key = ""
        # The URL for the Gemini API call
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

    def _load_json_file(self, file_path: Path):
        """
        Loads a JSON file from the specified path.

        Args:
            file_path (Path): The path to the JSON file.

        Returns:
            dict: The loaded data, or an empty dict if the file is not found or invalid.
        """
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                print(f"Warning: Knowledge base file not found at {file_path}. Initializing with empty data.")
                return {}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Initializing with empty data.")
            return {}
        except IOError as e:
            print(f"Error: Failed to read file at {file_path}. {e}")
            return {}

    async def interpret(self, phrase: str, context: str = None):
        """
        Grounds symbolic meaning based on the knowledge base and optional context.

        Args:
            phrase (str): The word or phrase to interpret.
            context (str): Optional situational or emotional context.

        Returns:
            dict: Grounded meaning, associations, and a confidence score.
        """
        phrase = phrase.lower().strip()
        meaning = self.symbol_kb.get(phrase, {})

        if context:
            # Use the LLM for nuanced interpretation if context is provided
            try:
                contextual_meaning = await self._contextualize_with_llm(meaning, context)
                # Merge the LLM-generated nuance with the base meaning
                meaning.update(contextual_meaning)
                meaning["context_used"] = context
            except Exception as e:
                print(f"Error during LLM contextualization: {e}. Using base meaning.")
        
        if not meaning:
            return {"meaning": "unknown", "associations": [], "confidence": 0.0}

        return meaning

    async def _contextualize_with_llm(self, base_meaning: dict, context: str):
        """
        Calls the LLM to provide a nuanced interpretation based on context.

        Args:
            base_meaning (dict): The base meaning from the static knowledge base.
            context (str): The situational context.
        
        Returns:
            dict: A dictionary with 'meaning_nuanced' and 'confidence' from the LLM.
        """
        # Construct a clear, descriptive prompt for the LLM
        prompt_text = (
            f"Given the phrase '{base_meaning.get('phrase', 'N/A')}', with a base meaning of "
            f"'{base_meaning.get('meaning', 'N/A')}' and associations like "
            f"{base_meaning.get('associations', [])}, and a situational context of "
            f"'{context}', provide a more nuanced interpretation. "
            "Respond with a JSON object containing a `meaning_nuanced` string and a "
            "`confidence` score (a number between 0.0 and 1.0)."
        )

        # Define the desired JSON schema for the LLM's response
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "meaning_nuanced": {"type": "STRING"},
                        "confidence": {"type": "NUMBER"}
                    },
                    "propertyOrdering": ["meaning_nuanced", "confidence"]
                }
            }
        }

        # Call the Gemini API with exponential backoff
        max_retries = 3
        for i in range(max_retries):
            try:
                response = await self._fetch_with_backoff(self.api_url, payload)
                result = await response.json()
                
                if response.ok and result.get('candidates'):
                    # The response is a string of JSON, so we need to parse it
                    json_string = result['candidates'][0]['content']['parts'][0]['text']
                    return json.loads(json_string)
                else:
                    error_message = result.get('error', {}).get('message', 'Unknown API error')
                    print(f"API Error: {error_message}")
                    raise Exception(error_message)

            except Exception as e:
                if i < max_retries - 1:
                    await asyncio.sleep(2 ** i)  # Exponential backoff
                else:
                    raise Exception(f"Failed to get LLM response after {max_retries} retries: {e}")
        
        return {"meaning_nuanced": "unknown", "confidence": 0.0}

    async def _fetch_with_backoff(self, url, payload):
        """A simple fetch wrapper with exponential backoff logic."""
        response = await fetch(url, {
            'method': 'POST',
            'headers': {'Content-Type': 'application/json'},
            'body': JSON.stringify(payload)
        })
        return response

    def map_metaphor(self, abstract_phrase: str, target_domain: str):
        """
        Attempts metaphorical mapping between abstract and concrete domains.

        Args:
            abstract_phrase (str): The abstract metaphor to map (e.g., "emotion is weather").
            target_domain (str): The specific term within the metaphor (e.g., "anger").

        Returns:
            dict: The mapped result.
        """
        mapping = self.metaphor_map.get(abstract_phrase.lower(), {})
        if not mapping:
            return {"mapped": False, "details": None}
        
        return {"mapped": True, "details": mapping.get(target_domain)}

    def register_new_symbol(self, phrase: str, meaning: str, associations: list = None, confidence: float = 0.7):
        """
        Allows for dynamically expanding the symbolic knowledge base.

        Args:
            phrase (str): The new word or phrase to add.
            meaning (str): The core meaning of the symbol.
            associations (list): A list of related concepts.
            confidence (float): The system's confidence in this meaning (0.0-1.0).
        """
        associations = associations or []
        entry = {
            "phrase": phrase.lower(),
            "meaning": meaning,
            "associations": associations,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        self.symbol_kb[phrase.lower()] = entry
        self._persist(self.symbol_kb, SYMBOL_KB_PATH)

    def _persist(self, data: dict, file_path: Path):
        """
        Saves the updated knowledge base to a file.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Error: Failed to save knowledge base to {file_path}. {e}")

# --- Test Driver ---
async def main():
    interpreter = SymbolInterpreter()

    # Demonstrating the LLM-powered interpretation
    phrase_to_interpret = "water"
    context_for_interpretation = "a dream of cleansing"
    print(f"Interpreting '{phrase_to_interpret}' with context '{context_for_interpretation}'...")
    interpretation = await interpreter.interpret(phrase_to_interpret, context=context_for_interpretation)
    print("Interpretation:", json.dumps(interpretation, indent=2))
    print("-" * 20)

    # Demonstrating metaphorical mapping
    print("Mapping metaphor 'emotion is weather' to 'calm'...")
    metaphor_result = interpreter.map_metaphor("emotion is weather", "calm")
    print("Metaphor Result:", metaphor_result)
    print("-" * 20)

    # Demonstrating registering a new symbol
    print("Registering new symbol 'phoenix'...")
    interpreter.register_new_symbol(
        phrase="phoenix",
        meaning="rebirth",
        associations=["fire", "cycle", "transformation"]
    )
    print("Symbol 'phoenix' registered. Check the symbolic_meanings.json file.")
    print("-" * 20)
    
if __name__ == "__main__":
    # Ensure the datasets directories exist for the test driver to work
    (KB_BASE_DIR / "datasets" / "symbols").mkdir(parents=True, exist_ok=True)
    (KB_BASE_DIR / "datasets" / "metaphors").mkdir(parents=True, exist_ok=True)
    # Create mock JSON files if they don't exist
    if not SYMBOL_KB_PATH.exists():
        with open(SYMBOL_KB_PATH, 'w') as f:
            json.dump({
                "light": {"phrase": "light", "meaning": "illumination", "associations": ["hope", "knowledge"], "confidence": 0.7},
                "water": {"phrase": "water", "meaning": "life-giving element", "associations": ["cleansing", "flow"], "confidence": 0.8}
            }, f, indent=2)
    if not METAPHOR_MAP_PATH.exists():
        with open(METAPHOR_MAP_PATH, 'w') as f:
            json.dump({
                "time is space": {"past": "behind", "future": "ahead"},
                "emotion is weather": {"anger": "storm", "calm": "clear sky"},
                "self is journey": {"growth": "path", "change": "turning point"}
            }, f, indent=2)
    
    # Run the main async function
    asyncio.run(main())


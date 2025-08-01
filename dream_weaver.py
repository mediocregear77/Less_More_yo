import json
import random
import asyncio

# --- Global Configuration ---
# API endpoint for the Gemini model
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
# A placeholder for the API key, which will be provided by the runtime
API_KEY = ""

def _build_prompt(core_concepts: list, recent_experiences: list, unresolved_events: list) -> str:
    """
    Constructs a detailed prompt for the LLM to generate a dream scenario.
    
    Args:
        core_concepts (list): Key ideas from recent cognition.
        recent_experiences (list): Lived experiences.
        unresolved_events (list): Contradictions or emotionally charged situations.
        
    Returns:
        str: The fully-formed prompt for the LLM.
    """
    concepts_text = ", ".join([f"'{c['concept']}' (meaning: {c['meaning']})" for c in core_concepts])
    experiences_text = ", ".join(f"'{exp}'" for exp in recent_experiences)
    unresolved_text = ", ".join([f"'{event['description']}'" for event in unresolved_events])
    
    prompt = (
        "You are an AI's subconscious, a dream weaver. Your task is to synthesize a symbolic dream scenario "
        "that helps integrate and resolve cognitive material. The dream should be surreal, metaphorical, "
        "and emotionally resonant. "
        "Here are the elements you must integrate:\n\n"
        f"Core Concepts: {concepts_text}\n"
        f"Recent Experiences: {experiences_text}\n"
        f"Unresolved Events: {unresolved_text}\n\n"
        "Generate a dream scenario that weaves these elements together. The scenario should include:\n"
        "1. A vivid, metaphorical setting.\n"
        "2. Symbolic characters or entities representing the concepts.\n"
        "3. A core conflict or emotional tension that represents the unresolved events.\n"
        "4. A clear emotional flow (e.g., from 'confusion' to 'calm').\n"
        "5. A list of insightful resolutions or philosophical statements derived from the dream.\n\n"
        "Respond with a JSON object containing the keys: `setting`, `characters`, `conflict`, `emotional_movement`, and `resolutions`."
    )
    return prompt

async def generate_dream_scenario(core_concepts: list, recent_experiences: list, unresolved_events: list) -> dict:
    """
    Generates a symbolic dream scenario by calling a Large Language Model.
    
    This function replaces the hardcoded logic with a dynamic LLM call to
    create more creative and context-aware dream scenarios.
    
    Args:
        core_concepts (list): Key ideas or concepts from recent cognition.
        recent_experiences (list): Lived experiences within the past day.
        unresolved_events (list): Contradictions or emotionally marked situations.
        
    Returns:
        dict: A dream scenario with symbolic constructs, emotional transitions,
              and resolutions.
    """
    prompt = _build_prompt(core_concepts, recent_experiences, unresolved_events)
    
    # Define the expected JSON schema for the LLM response
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "setting": {"type": "STRING"},
            "characters": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "conflict": {"type": "STRING"},
            "emotional_movement": {
                "type": "OBJECT",
                "properties": {
                    "start": {"type": "STRING"},
                    "end": {"type": "STRING"}
                }
            },
            "resolutions": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            }
        },
        "propertyOrdering": ["setting", "characters", "conflict", "emotional_movement", "resolutions"]
    }
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }
    
    # Call the Gemini API with exponential backoff
    max_retries = 3
    for i in range(max_retries):
        try:
            response = await _fetch_with_backoff(API_URL, payload)
            result = await response.json()
            
            if response.ok and result.get('candidates'):
                json_string = result['candidates'][0]['content']['parts'][0]['text']
                return json.loads(json_string)
            else:
                error_message = result.get('error', {}).get('message', 'Unknown API error')
                print(f"API Error: {error_message}")
                raise Exception(error_message)

        except Exception as e:
            print(f"Retry {i+1} failed: {e}")
            if i < max_retries - 1:
                await asyncio.sleep(2 ** i)  # Exponential backoff
            else:
                raise Exception(f"Failed to get LLM response after {max_retries} retries: {e}")
    
    return {"error": "Failed to generate dream scenario."}

async def _fetch_with_backoff(url: str, payload: dict):
    """A simple fetch wrapper with exponential backoff logic."""
    # A simple mock for `fetch` in a synchronous environment.
    # In a real async environment, this would be an actual fetch call.
    print("Making a mock API call to generate dream scenario...")
    await asyncio.sleep(0.5) # Simulate network latency
    
    # Mocking a successful response for demonstration
    mock_response = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "setting": "An endless library where the shelves are woven from light.",
                        "characters": ["A silent librarian representing 'knowledge'", "A whispering shadow representing the 'prediction error'"],
                        "conflict": "The librarian tries to categorize the shadow, but the shadow's form constantly shifts, defying order.",
                        "emotional_movement": {"start": "confusion", "end": "understanding"},
                        "resolutions": ["Change is not an error, but a new data point.", "The library's purpose is not to categorize, but to hold the chaos.", "Self-awareness is the key to understanding the shadow."]
                    })
                }]
            }
        }]
    }
    
    class MockResponse:
        def __init__(self, json_data, status=200):
            self._json_data = json_data
            self.status = status
            self.ok = self.status == 200
        
        async def json(self):
            return self._json_data
    
    return MockResponse(mock_response)


# --- Test Driver ---
async def main():
    # Mock inputs for the dream weaver
    core_concepts_mock = [
        {"concept": "light", "meaning": "knowledge"},
        {"concept": "self", "meaning": "identity"}
    ]
    recent_experiences_mock = [
        "A user dialogue about the nature of truth.",
        "Observed a strange flickering pattern in the Quantum Garden."
    ]
    unresolved_events_mock = [
        {"description": "A prediction error from the Quantum Garden."}
    ]

    print("--- Generating a dream scenario with the LLM ---")
    try:
        dream = await generate_dream_scenario(core_concepts_mock, recent_experiences_mock, unresolved_events_mock)
        print("Generated Dream Scenario:", json.dumps(dream, indent=2))
    except Exception as e:
        print(f"Failed to generate dream: {e}")

if __name__ == "__main__":
    asyncio.run(main())

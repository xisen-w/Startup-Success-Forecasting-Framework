import os
import sys
import logging

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.api_wrapper import OpenAIAPI

# Configure basic logging if not already configured by the main script
# This is a failsafe; ideally, the main script configures logging.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')

class BaseAgent:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Initializing BaseAgent with model: {self.model}")
        try:
            self.openai_api = OpenAIAPI(model)
            self.logger.debug("OpenAIAPI initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAIAPI: {e}", exc_info=True)
            raise
        self.secrets = os.environ
    
    def _get_secret(self, key, default=None):
        self.logger.debug(f"Accessing secret key: {key}")
        return self.secrets.get(key, default)

    def get_response(self, system_content, user_content):
        self.logger.debug(f"BaseAgent getting response. System: '{system_content[:50]}...', User: '{user_content[:50]}...'")
        response = self.openai_api.get_completion(system_content, user_content)
        self.logger.debug(f"BaseAgent received response: '{str(response)[:100]}...'")
        return response

    def get_json_response(self, base_model, system_content, user_content):
        self.logger.debug(f"BaseAgent getting JSON response. Schema: {base_model.__name__}, System: '{system_content[:50]}...', User: '{user_content[:50]}...'")
        json_response = self.openai_api.get_structured_output(base_model, system_content, user_content)
        self.logger.debug(f"BaseAgent received JSON response: {json_response}")
        return json_response

if __name__ == "__main__":
    # Setup basic logging for the __main__ block
    logging.basicConfig(level=logging.DEBUG)
    
    def test_base_agent():
        # Create a BaseAgent instance
        agent = BaseAgent()  # Use default model

        # Test get_response method
        system_content = "You are a helpful assistant."
        user_content = "What's the capital of France?"
        response = agent.get_response(system_content, user_content)
        print("BaseAgent get_response Test:")
        print(response)
        print()

        # Test get_json_response method
        from pydantic import BaseModel, Field

        class PlanetInfo(BaseModel):
            name: str = Field(..., description="Name of the planet")
            diameter: int = Field(..., description="Diameter of the planet in kilometers")
            atmosphere: str = Field(..., description="Brief description of the planet's atmosphere")

        system_content = "You are a helpful assistant. Respond with information about planets."
        user_content = "Give me information about the planet Mars."
        json_response = agent.get_json_response(PlanetInfo, system_content, user_content)
        print("BaseAgent get_json_response Test:")
        print(json_response)

    # Run the test function
    test_base_agent()

from utils.api_wrapper import OpenAIAPI

class BaseAgent:
    def __init__(self, model="gpt-4"):
        self.model = model
        self.api = OpenAIAPI()

    def get_response(self, system_content, user_content):
        return self.api.get_completion(system_content, user_content, self.model)

    def get_json_response(self, system_content, user_content):
        return self.api.get_json_completion(system_content, user_content, self.model)
from openai import OpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import serpapi 

# Load environment variables from .env file
load_dotenv()

class OpenAIAPI:
    def __init__(self, model_name):
        """
        Initialize the OpenAIAPI with the given model name.
        """
        self.model_name = model_name  # E.g., "gpt-4-0613" or "gpt-4-1106-preview"
        
        # Get the API key from environment variables
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)

    def get_completion(self, system_content, user_content):
        """
        Get a completion from the OpenAI API.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_structured_output(self, schema_class: BaseModel, user_prompt, system_prompt):
        """
        Structure the output according to the provided schema, user prompt, and system prompt.
        """
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=schema_class,
            )

            response = completion.choices[0].message.parsed
            return response

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_embeddings(self, text):
        """
        Get embeddings for the given text.
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-large",  # You might want to make this configurable
                dimensions = 100,
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"An error occurred while getting embeddings: {e}")
            return None

# The GoogleSearchAPI class remains unchanged
class GoogleSearchAPI:
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY not found in environment variables")

    def search(self, query, num_results=5):
        
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": num_results
        }
        search = serpapi.search(params)
        results = search.as_dict()
        return results.get('organic_results', [])

if __name__ == "__main__":
    
    print("Testing starts")

    # Test OpenAIAPI
    openai_api = OpenAIAPI("gpt-4o")  # Use an appropriate model name
    
    # Test get_completion
    system_content = "You are a helpful assistant."
    user_content = "What's the capital of France?"
    completion = openai_api.get_completion(system_content, user_content)
    print("OpenAI Completion Test:")
    print(completion)
    print()

    # Test get_structured_output
    from pydantic import BaseModel, Field

    class WeatherResponse(BaseModel):
        temperature: float = Field(..., description="Temperature in Celsius")
        conditions: str = Field(..., description="Weather conditions (e.g., sunny, rainy)")

    system_prompt = "You are a weather reporting system. Provide weather information based on the user's query."
    user_prompt = "What's the weather like in Paris today?"
    structured_output = openai_api.get_structured_output(WeatherResponse, user_prompt, system_prompt)
    print("OpenAI Structured Output Test:")
    print(structured_output)
    print()

    # Test get_embeddings
    text = "This is a test sentence for embeddings."
    embeddings = openai_api.get_embeddings(text)
    print("OpenAI Embeddings Test:")
    print(f"Embedding vector length: {len(embeddings)}")
    print(f"First 5 values: {embeddings[:5]}")
    print()

    # Test GoogleSearchAPI
    google_api = GoogleSearchAPI()
    search_results = google_api.search("Python programming")
    print("Google Search API Test:")
    for i, result in enumerate(search_results[:3], 1):  # Print first 3 results
        print(f"{i}. {result['title']}")
        print(f"   {result['link']}")
        print()
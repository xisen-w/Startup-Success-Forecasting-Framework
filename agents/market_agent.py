import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from agents.base_agent import BaseAgent
from utils.api_wrapper import GoogleSearchAPI
from pydantic import BaseModel, Field

class MarketAnalysis(BaseModel):
    market_size: str = Field(..., description="Estimated market size")
    growth_rate: str = Field(..., description="Market growth rate")
    competition: str = Field(..., description="Overview of competition")
    market_trends: str = Field(..., description="Key market trends")
    viability_score: int = Field(..., description="Market viability score on a scale of 1 to 10")

class MarketAgent(BaseAgent):
    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model)
        self.search_api = GoogleSearchAPI()

    def analyze(self, startup_info, mode):
        market_info = self._get_market_info(startup_info)
        analysis = self.get_json_response(MarketAnalysis, self._get_analysis_prompt(), market_info)
        
        if mode == "advanced":
            external_knowledge = self._get_external_knowledge(startup_info)
            advanced_analysis = self.get_json_response(MarketAnalysis, self._get_advanced_analysis_prompt(), f"{market_info}\n\nAdditional Information:\n{external_knowledge}")
            return advanced_analysis
        
        return analysis

    def _get_market_info(self, startup_info):
        return f"Market size: {startup_info.get('market_size', '')}\n" \
               f"Competition: {startup_info.get('competition', '')}\n" \
               f"Market Growth Rate: {startup_info.get('growth_rate', '')}\n" \
               f"Market Trends: {startup_info.get('market_trends', '')}"

    def _get_external_knowledge(self, startup_info):
        keywords = self._generate_keywords(startup_info)
        search_results = self.search_api.search(keywords)
        return self._synthesize_knowledge(search_results)

    def _generate_keywords(self, startup_info):
        prompt = f"Generate 3-5 search keywords based on this startup information: {startup_info['description']}"
        return self.get_response(self._get_keyword_generation_prompt(), prompt)

    def _synthesize_knowledge(self, search_results):
        synthesis_prompt = f"Synthesize the following search results into a concise market overview:\n\n{search_results}"
        return self.get_response(self._get_synthesis_prompt(), synthesis_prompt)

    def _get_analysis_prompt(self):
        return """
        As an experienced market analyst, analyze the startup's market based on the following information:
        {market_info}

        Provide a comprehensive analysis including market size, growth rate, competition, and key trends.
        Conclude with a market viability score from 1 to 10.
        """

    def _get_advanced_analysis_prompt(self):
        return """
        As an experienced market analyst, provide an in-depth analysis of the startup's market based on the following information:
        {market_info}

        Include insights from the additional external research provided.
        Provide a comprehensive analysis including market size, growth rate, competition, and key trends.
        Conclude with a market viability score from 1 to 10, factoring in the external data.
        """

    def _get_keyword_generation_prompt(self):
        return "You are an AI assistant skilled at generating relevant search keywords. Please provide 3-5 concise keywords or short phrases based on the given information."

    def _get_synthesis_prompt(self):
        return "You are an AI assistant skilled at synthesizing information. Please provide a concise summary of the key points from the given search results, focusing on market trends, size, and competitive landscape."

if __name__ == "__main__":
    def test_market_agent():
        # Create a MarketAgent instance
        agent = MarketAgent()

        # Test startup info
        startup_info = {
            "description": "AI-powered health monitoring wearable device",
            "market_size": "$50 billion global wearable technology market",
            "competition": "Fitbit, Apple Watch, Garmin",
            "growth_rate": "CAGR of 15.9% from 2020 to 2027",
            "market_trends": "Increasing health consciousness, integration of AI in healthcare"
        }

        # Test basic analysis
        print("Basic Analysis:")
        basic_analysis = agent.analyze(startup_info, mode="basic")
        print(basic_analysis)
        print()

        # Test advanced analysis
        print("Advanced Analysis:")
        advanced_analysis = agent.analyze(startup_info, mode="advanced")
        print(advanced_analysis)

    # Run the test function
    test_market_agent()
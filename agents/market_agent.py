from agents.base_agent import BaseAgent
from utils.api_wrapper import GoogleSearchAPI

class MarketAgent(BaseAgent):
    def __init__(self, model="gpt-4"):
        super().__init__(model)
        self.search_api = GoogleSearchAPI()

    def analyze(self, startup_info, mode):
        market_info = self._get_market_info(startup_info)
        if mode == "advanced":
            external_knowledge = self._get_external_knowledge(startup_info)
            market_info += f"\n{external_knowledge}"
        return self.get_response(self._get_analysis_prompt(), market_info)

    def _get_market_info(self, startup_info):
        # Extract relevant market information from startup_info
        return f"Market size: {startup_info.get('market_size', '')}\n" \
               f"Competition: {startup_info.get('competition', '')}\n" \
               f"Market Growth Rate: {startup_info.get('growth_rate', '')}\n" \
               f"Market Trends: {startup_info.get('market_trends', '')}"

    def _get_external_knowledge(self, startup_info):
        keywords = self._generate_keywords(startup_info)
        search_results = self.search_api.search(keywords)
        return self._synthesize_knowledge(search_results)

    def _generate_keywords(self, startup_info):
        # Generate search keywords based on startup info
        pass

    def _synthesize_knowledge(self, search_results):
        # Synthesize external knowledge from search results
        pass

    def _get_analysis_prompt(self):
        return """
        As an experienced market analyst, analyze the startup based on the following information:
        {market_info}

        Consider the current market size, projected growth rate, existing competition, and prevailing market trends.
        Provide a well-reasoned analysis and conclude with a market viability score from 1 to 10.
        """
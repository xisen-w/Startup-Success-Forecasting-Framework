import os
import sys
import logging

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
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def analyze(self, startup_info, mode):
        self.logger.info(f"Starting market analysis in {mode} mode")
        market_info = self._get_market_info(startup_info)
        self.logger.debug(f"Market info: {market_info}")
        
        analysis = self.get_json_response(MarketAnalysis, self._get_analysis_prompt(), market_info)
        self.logger.info("Basic analysis completed")
        
        if mode == "advanced":
            self.logger.info("Starting advanced analysis")
            external_knowledge = self._get_external_knowledge(startup_info)
            self.logger.debug(f"External knowledge: {external_knowledge}")
            advanced_analysis = self.get_json_response(MarketAnalysis, self._get_advanced_analysis_prompt(), f"{market_info}\n\nAdditional Information:\n{external_knowledge}")
            self.logger.info("Advanced analysis completed")
            return advanced_analysis
        
        if mode == "natural_language_advanced":
            self.logger.info("Starting advanced analysis")

            # Generate and log keywords
            keywords = self._generate_keywords(startup_info)
            print("\nSearch Keywords Generated:")
            print("-" * 40)
            print(keywords)
            
            # Log raw search results before synthesis
            print("\nRaw Search Results:")
            print("-" * 40)
            search_results = self.search_api.search(keywords)  # Raw search results
            print(search_results)
            
            # Log synthesized knowledge
            print("\nSynthesized External Knowledge:")
            print("-" * 40)
            synthesized_knowledge = self._synthesize_knowledge(search_results)
            print(synthesized_knowledge)

            # Get external knowledge
            external_knowledge = self._get_external_knowledge(startup_info)
            
            prompt = self.natural_language_analysis_prompt().format(
                startup_info=startup_info,
                market_info=market_info,
                keywords=keywords,
                external_knowledge="Knowledge 1: " + external_knowledge + "\n" + "Knowledge 2: " + synthesized_knowledge
            )
            
            nl_advanced_analysis = self.get_response(prompt, "Formulate a professional and comprehensive analysis please.")
            self.logger.info("Natural language analysis completed")
            return {
                'analysis': nl_advanced_analysis,
                'external_report': external_knowledge
            }
        
        return analysis

    def _get_market_info(self, startup_info):
        return f"Market size: {startup_info.get('market_size', '')}\n" \
               f"Competition: {startup_info.get('competition', '')}\n" \
               f"Market Growth Rate: {startup_info.get('growth_rate', '')}\n" \
               f"Market Trends: {startup_info.get('market_trends', '')}"

    def _get_external_knowledge(self, startup_info):
        """Get structured market report from external sources"""
        self.logger.info("Starting external knowledge gathering")
        
        # Log keyword generation
        keywords = self._generate_keywords(startup_info)
        self.logger.info(f"Generated keywords: {keywords}")
        
        # Log search API call
        search_results = self.search_api.search(keywords)
        self.logger.info("Raw search results received")
        
        # Log organic results details  
        print("\nORGANIC RESULTS:")
        print("-" * 40)
        if isinstance(search_results, list):  # Direct list of results
            organic_results = search_results[:20]
            print(f"Number of organic results: {len(organic_results)}")
            for i, result in enumerate(organic_results):
                print(f"\nResult {i+1}:")
                print(f"Source: {result.get('source', 'No source')}")
                print(f"Title: {result.get('title', 'No title')}")
                print(f"Date: {result.get('date', 'No date')}")
                print(f"Snippet: {result.get('snippet', 'No snippet')}")
        
        # Compile structured knowledge
        overall_knowledge = "Market Research Summary:\n\n"
        if isinstance(search_results, list):
            for result in search_results[:20]:
                source = result.get('source', '')
                date = result.get('date', '')
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                
                # Only add entries that have actual content
                if snippet:
                    overall_knowledge += f"Source: {source} ({date})\n"
                    overall_knowledge += f"Title: {title}\n"
                    overall_knowledge += f"Finding: {snippet}\n\n"
        
        print("\nSTRUCTURED KNOWLEDGE:")
        print("-" * 40)
        print(overall_knowledge)
        
        synthesis_prompt = """As a market research analyst, synthesize the following market data into a structured report.
        Focus on:
        1. Market size and growth rates (include specific numbers)
        2. Industry trends and developments
        3. Competitive dynamics
        4. Market timing and sentiment
        
        Use specific data points from the research where available.
        Format your response as a clear, data-driven market report."""
        
        market_report = self.get_response(synthesis_prompt, overall_knowledge)
        print("\nFINAL MARKET REPORT:")
        print("-" * 40)
        print(market_report)
        
        return market_report

    def _generate_keywords(self, startup_info):
        """Generate focused market keywords for research"""
        keyword_prompt = ("You will assist me in finding external market knowledge about a startup. Think step by step. "
                         "Your task is to summarise the information into 1 keyword that best describes the market that the startup is in. "
                         "Sample Output: Chinese Pharmaceutical Market.")
        main_keyword = self.get_response(keyword_prompt, startup_info['description'])
        return f"{main_keyword}, Growth, Trend, Size, Revenue"

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
    
    def natural_language_analysis_prompt(self):
        return """
        You are a professional agent in a VC firm to analyze a company. Your task is to analyze the company here. Context: {startup_info}

        Your focus is on the market side. What is the market? Is the market big enough? Is now the good timing? Will there be a good product-market-fit? 
        
        Specifically here are some relevant market information: {market_info}. 

        Your intern has researched more around the following topic for you as context {keywords}.

        The research result: {external_knowledge}

        Provide a comprehensive analysis including market size, growth rate, competition, and key trends. Analyze step by step to formulate your comprehensive analysis to answer the questions proposed above.

        Also conclude with a market viability score from 1 to 10. 
        """

    def _get_keyword_generation_prompt(self):
        return "You are an AI assistant skilled at generating relevant search keywords. Please provide 3-5 concise keywords or short phrases based on the given information."

    def _get_synthesis_prompt(self):
        return """
    You are a market research analyst. Synthesize the search results focusing on quantitative data points:
    
    - Market size (in USD)
    - Growth rates (CAGR)
    - Market share percentages
    - Transaction volumes
    - Customer acquisition costs
    - Revenue metrics
    - Competitive landscape metrics
    
    Format data points clearly and cite their time periods. If exact numbers aren't available, 
    provide ranges based on available data. Prioritize numerical data over qualitative descriptions.
    """

if __name__ == "__main__":
    def test_market_agent():
        # Configure logging for the test function
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)

        logger.info("Starting MarketAgent test")
        agent = MarketAgent()

        # Test startup info based on Stripe's early days (circa 2010-2011)
        startup_info = {
            "description": "Developer-first payment processing platform that allows businesses to accept and manage online payments through simple API integration. The platform handles everything from payment acceptance to fraud prevention, banking infrastructure, and business analytics.",
            "market_size": "Global digital payments market valued at approximately $1.2 trillion in 2010",
            "competition": "PayPal, Square, traditional payment processors (First Data, Chase Paymentech), and legacy banking systems",
            "growth_rate": "Digital payments market CAGR of 20% from 2010 to 2015, with accelerating adoption of online commerce",
            "market_trends": """
            - Rapid shift from brick-and-mortar to online commerce
            - Growing demand for developer-friendly payment solutions
            - Increasing focus on mobile payments and digital wallets
            - Rising need for cross-border payment solutions
            - Emergence of platform business models requiring complex payment flows
            """
        }

        print("\n=== Starting Analysis of Stripe (2010-2011 perspective) ===")
        print("-" * 80)
        
        # Log the generated keywords
        keywords = agent._generate_keywords(startup_info)
        print("\nGenerated Search Keywords:")
        print("-" * 40)
        print(keywords)
        
        # Log the external knowledge gathered
        external_knowledge = agent._get_external_knowledge(startup_info)
        print("\nExternal Market Research:")
        print("-" * 40)
        print(external_knowledge)
        
        # Perform and log the full analysis
        print("\nFull Market Analysis:")
        print("-" * 40)
        nl_analysis = agent.analyze(startup_info, mode="natural_language_advanced")
        print(nl_analysis)

        print("\n=== Raw Search Data Collection ===")
        print("-" * 80)
        
        # Generate and log keywords
        keywords = agent._generate_keywords(startup_info)
        print("\nSearch Keywords Generated:")
        print("-" * 40)
        print(keywords)
        
        # Log raw search results before synthesis
        print("\nRaw Search Results:")
        print("-" * 40)
        search_results = agent.search_api.search(keywords)  # Raw search results
        print(search_results)
        
        # Log synthesized knowledge
        print("\nSynthesized External Knowledge:")
        print("-" * 40)
        synthesized_knowledge = agent._synthesize_knowledge(search_results)
        print(synthesized_knowledge)

    # Run the test function
    test_market_agent()
import os
import sys
import logging

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from agents.base_agent import BaseAgent
from pydantic import BaseModel, Field

class ProductAnalysis(BaseModel):
    features_analysis: str = Field(..., description="Analysis of product features")
    tech_stack_evaluation: str = Field(..., description="Evaluation of the technology stack")
    usp_assessment: str = Field(..., description="Assessment of the unique selling proposition")
    potential_score: int = Field(..., description="Product potential score on a scale of 1 to 10")
    innovation_score: int = Field(..., description="Innovation score on a scale of 1 to 10")
    market_fit_score: int = Field(..., description="Market fit score on a scale of 1 to 10")

class ProductAgent(BaseAgent):
    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def analyze(self, startup_info, mode):
        self.logger.info(f"Starting product analysis in {mode} mode")
        product_info = self._get_product_info(startup_info)
        self.logger.debug(f"Product info: {product_info}")
        
        if not product_info.strip():
            self.logger.warning("No product information available")
            return ProductAnalysis(
                features_analysis="No information available",
                tech_stack_evaluation="No information available",
                usp_assessment="No information available",
                potential_score=0,
                innovation_score=0,
                market_fit_score=0
            )
        
        analysis = self.get_json_response(ProductAnalysis, self._get_analysis_prompt(), product_info)
        self.logger.info("Basic analysis completed")
        
        if mode == "advanced":
            self.logger.info("Starting advanced analysis")
            innovation_score = self._assess_innovation(product_info)
            market_fit = self._assess_market_fit(startup_info, product_info)
            analysis.innovation_score = innovation_score
            analysis.market_fit_score = market_fit
            self.logger.info("Advanced analysis completed")
        
        return analysis

    def _get_product_info(self, startup_info):
        return f"Product Description: {startup_info.get('product_details', '')}\n" \
               f"Key Features: {startup_info.get('product_details', '')}\n" \
               f"Technology Stack: {startup_info.get('technology_stack', '')}\n" \
               f"Unique Selling Proposition: {startup_info.get('product_fit', '')}"

    def _assess_innovation(self, product_info):
        self.logger.info("Assessing innovation")
        innovation_analysis = self.get_response(self._get_innovation_prompt(), product_info)
        self.logger.debug(f"Innovation analysis: {innovation_analysis}")
        return int(innovation_analysis.split('\n')[0])  # Assuming the score is the first line

    def _assess_market_fit(self, startup_info, product_info):
        self.logger.info("Assessing market fit")
        combined_info = f"{startup_info.get('market_description', '')}\n{product_info}"
        market_fit_analysis = self.get_response(self._get_market_fit_prompt(), combined_info)
        self.logger.debug(f"Market fit analysis: {market_fit_analysis}")
        return int(market_fit_analysis.split('\n')[0])  # Assuming the score is the first line

    def _get_analysis_prompt(self):
        return """
        As a product expert, analyze the startup's product based on the following information:
        {product_info}

        Consider the product's features, technology stack, and unique selling proposition.
        Provide a comprehensive analysis and rate the product's potential on a scale of 1 to 10.
        """

    def _get_innovation_prompt(self):
        return """
        Assess the level of innovation in the product based on the following information:
        {product_info}

        Rate the innovation on a scale of 1 to 10, where 1 is not innovative at all and 10 is groundbreaking innovation.
        Provide the rating as a single number on the first line, followed by a brief explanation for your rating.
        """

    def _get_market_fit_prompt(self):
        return """
        Evaluate the product-market fit based on the following information about the market and the product:
        {combined_info}

        Rate the product-market fit on a scale of 1 to 10, where 1 is poor fit and 10 is perfect fit.
        Provide the rating as a single number on the first line, followed by a brief explanation for your rating.
        """

if __name__ == "__main__":
    def test_product_agent():
        # Configure logging for the test function
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info("Starting ProductAgent test")
        # Create a ProductAgent instance
        agent = ProductAgent()

        # Test startup info
        startup_info = {
            "product_description": "AI-powered health monitoring wearable device",
            "key_features": "Real-time health tracking, Personalized AI insights, Integration with medical systems",
            "tech_stack": "IoT sensors, Machine Learning algorithms, Cloud computing",
            "usp": "Predictive health analysis with medical-grade accuracy",
            "market_description": "Growing health tech market with increasing demand for personalized healthcare solutions"
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
    test_product_agent()
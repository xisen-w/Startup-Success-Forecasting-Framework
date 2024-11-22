import logging
from typing import Dict, Any
from pydantic import BaseModel

from agents.market_agent import MarketAgent
from agents.product_agent import ProductAgent
from agents.founder_agent import FounderAgent
from agents.vc_scout_agent import VCScoutAgent, StartupInfo
from agents.integration_agent import IntegrationAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StartupFramework:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.market_agent = MarketAgent(model)
        self.product_agent = ProductAgent(model)
        self.founder_agent = FounderAgent(model)
        self.vc_scout_agent = VCScoutAgent(model)
        self.integration_agent = IntegrationAgent(model)

    def analyze_startup(self, startup_info_str: str) -> Dict[str, Any]:
        logger.info("Starting startup analysis in advanced mode")

        # Parse the input string into a StartupInfo schema
        startup_info = self.vc_scout_agent.parse_record(startup_info_str)

        print("Parse Record: ", startup_info)

        # Check if parsing was successful
        if isinstance(startup_info, dict):
            startup_info = StartupInfo(**startup_info)
        elif not isinstance(startup_info, StartupInfo):
            logger.error("Failed to parse startup info")
            return {"error": "Failed to parse startup info"}

        # Get prediction and categorization
        prediction, categorization = self.vc_scout_agent.side_evaluate(startup_info)
        logger.info(f"VCScout prediction: {prediction}")

        # Perform agent analyses
        market_analysis = self.market_agent.analyze(startup_info.dict(), "advanced")
        product_analysis = self.product_agent.analyze(startup_info.dict(), "advanced")
        founder_analysis = self.founder_agent.analyze(startup_info.dict(), "advanced")

        # Log the startup_info for debugging
        logger.debug(f"Startup info: {startup_info.dict()}")

        founder_segmentation = self.founder_agent.segment_founder(startup_info.founder_backgrounds)
        founder_idea_fit = self.founder_agent.calculate_idea_fit(startup_info.dict(), startup_info.founder_backgrounds)

        # Integrate analyses
        integrated_analysis = self.integration_agent.integrated_analysis_pro(
            market_info=market_analysis.dict(),
            product_info=product_analysis.dict(),
            founder_info=founder_analysis.dict(),  
            founder_idea_fit=founder_idea_fit,
            founder_segmentation=founder_segmentation,
            rf_prediction=prediction,
        )

        quant_decision = self.integration_agent.getquantDecision(
            prediction,
            founder_idea_fit,
            founder_segmentation,
        )

        return {
            'Final Analysis': integrated_analysis.dict(),
            'Market Analysis': market_analysis.dict(),
            'Product Analysis': product_analysis.dict(),
            'Founder Analysis': founder_analysis.dict(),
            'Founder Segmentation': founder_segmentation,
            'Founder Idea Fit': founder_idea_fit[0],
            'Categorical Prediction': prediction,
            'Categorization': categorization.dict(),
            'Quantitative Decision': quant_decision.dict(),
            'Startup Info': startup_info.dict()
        }

    def analyze_startup_natural(self, startup_info_str: str) -> Dict[str, Any]:
        """Analyze startup using natural language processing mode"""
        logger.info("Starting startup analysis in natural language mode")

        # Parse the input string into a StartupInfo schema
        startup_info = self.vc_scout_agent.parse_record(startup_info_str)

        print("Parse Record: ", startup_info)

        # Check if parsing was successful
        if isinstance(startup_info, dict):
            startup_info = StartupInfo(**startup_info)
        elif not isinstance(startup_info, StartupInfo):
            logger.error("Failed to parse startup info")
            return {"error": "Failed to parse startup info"}

        # Get prediction and categorization
        prediction, categorization = self.vc_scout_agent.side_evaluate(startup_info)
        logger.info(f"VCScout prediction: {prediction}")

        # Perform agent analyses using natural language mode
        market_analysis = self.market_agent.analyze(startup_info.dict(), "natural_language_advanced")
        product_analysis = self.product_agent.analyze(startup_info.dict(), "natural_language_advanced")
        founder_analysis = self.founder_agent.analyze(startup_info.dict(), "advanced")  # Keep founder analysis in advanced mode

        # Log the analyses for debugging
        logger.debug(f"Market Analysis: {market_analysis}")
        logger.debug(f"Product Analysis: {product_analysis}")
        logger.debug(f"Founder Analysis: {founder_analysis}")

        # Get founder specific metrics
        founder_segmentation = self.founder_agent.segment_founder(startup_info.founder_backgrounds)
        founder_idea_fit = self.founder_agent.calculate_idea_fit(startup_info.dict(), startup_info.founder_backgrounds)

        # Integrate analyses
        integrated_analysis = self.integration_agent.integrated_analysis_pro(
            market_info={"analysis": market_analysis},  # Wrap in dict to maintain compatibility
            product_info={"analysis": product_analysis},
            founder_info=founder_analysis.dict(),
            founder_idea_fit=founder_idea_fit,
            founder_segmentation=founder_segmentation,
            rf_prediction=prediction,
        )

        quant_decision = self.integration_agent.getquantDecision(
            prediction,
            founder_idea_fit,
            founder_segmentation,
        )

        return {
            'Final Analysis': integrated_analysis.dict(),
            'Market Analysis': market_analysis,  # Direct natural language output
            'Product Analysis': product_analysis,  # Direct natural language output
            'Founder Analysis': founder_analysis.dict(),
            'Founder Segmentation': founder_segmentation,
            'Founder Idea Fit': founder_idea_fit[0],
            'Categorical Prediction': prediction,
            'Categorization': categorization.dict(),
            'Quantitative Decision': quant_decision.dict(),
            'Startup Info': startup_info.dict()
        }

def main():
    framework = StartupFramework()
    
    # Test case: Stripe (as an early-stage startup)
    startup_info_str = """
    Company: Stripe
    Description: Developer-first payment processing platform that allows businesses to accept and manage online payments through simple API integration.
    
    Market Information:
    - Global digital payments market valued at approximately $1.2 trillion
    - Growing at CAGR of 20% with accelerating adoption of online commerce
    - Key competitors include PayPal, Square, traditional payment processors
    
    Product Details:
    - RESTful API for payment processing
    - Support for 135+ currencies
    - Fraud prevention system
    - Real-time reporting dashboard
    - Subscription management
    - Invoice automation
    - Mobile SDK
    
    Technology Stack:
    - Ruby on Rails backend
    - React.js frontend
    - PostgreSQL database
    - Redis for caching
    - AWS infrastructure
    - Machine learning for fraud detection
    
    Founder Backgrounds:
    Patrick Collison: Previously founded and sold Auctomatic for $5M at age 19, degree in mathematics
    John Collison: Youngest self-made billionaire, studied physics at Harvard before dropping out
    
    Funding: Raised $2M in seed funding
    """

    print("\n=== Testing Natural Language Analysis ===")
    print("-" * 80)
    
    try:
        # Run natural language analysis
        print("\nStarting Natural Language Analysis...")
        natural_result = framework.analyze_startup_natural(startup_info_str)
        
        # Print results in a structured way
        print("\nNATURAL LANGUAGE ANALYSIS RESULTS:")
        print("-" * 40)
        
        print("\n1. MARKET ANALYSIS:")
        print("-" * 20)
        print(natural_result['Market Analysis'])
        
        print("\n2. PRODUCT ANALYSIS:")
        print("-" * 20)
        print(natural_result['Product Analysis'])
        
        print("\n3. FOUNDER ANALYSIS:")
        print("-" * 20)
        print(natural_result['Founder Analysis'].analysis)
        
        print("\n4. FINAL INTEGRATED ANALYSIS:")
        print("-" * 20)
        print(natural_result['Final Analysis'])
        
        print("\n5. QUANTITATIVE METRICS:")
        print("-" * 20)
        print(f"Founder Idea Fit: {natural_result['Founder Idea Fit']}")
        print(f"Categorical Prediction: {natural_result['Categorical Prediction']}")
        print(f"Quantitative Decision: {natural_result['Quantitative Decision']}")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

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

        # Integrate analyses
        integrated_analysis_basic = self.integration_agent.integrated_analysis_basic(
            market_info=market_analysis.dict(),
            product_info=product_analysis.dict(),
            founder_info=founder_analysis.dict(),  
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
            'Startup Info': startup_info.dict(),
            'Basic Analysis': integrated_analysis_basic.dict(),
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

        integrated_analysis_basic = self.integration_agent.integrated_analysis_basic(
            market_info={"analysis": market_analysis},  # Wrap in dict to maintain compatibility
            product_info={"analysis": product_analysis},
            founder_info=founder_analysis.dict(),
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
            'Startup Info': startup_info.dict(),
            'Basic Analysis': integrated_analysis_basic.dict(),
        }

def main():
    framework = StartupFramework("gpt-4o")
    
    # Test case: Turismocity (as an example)
    startup_info_str = """
    Turismocity is a travel search engine for Latin America that provides price comparison tools and travel deals. Eugenio Fage, the CTO and co-founder, has a background in software engineering and extensive experience in developing travel technology solutions.
    """

    print("\n=== Testing Advanced Analysis (analyze_startup) ===")
    print("-" * 80)
    
    try:
        print("\nStarting Advanced Analysis...")
        advanced_result = framework.analyze_startup(startup_info_str)
        
        print("\nADVANCED ANALYSIS RESULTS:")
        print("-" * 40)
        
        print("\n1. MARKET ANALYSIS:")
        print("-" * 20)
        print(advanced_result.get('Market Analysis', 'N/A'))
        
        print("\n2. PRODUCT ANALYSIS:")
        print("-" * 20)
        print(advanced_result.get('Product Analysis', 'N/A'))
        
        print("\n3. FOUNDER ANALYSIS:")
        print("-" * 20)
        print(advanced_result.get('Founder Analysis', 'N/A'))
        
        print("\n4. FINAL INTEGRATED ANALYSIS (PRO):")
        print("-" * 20)
        print(advanced_result.get('Final Analysis', 'N/A'))

        print("\n5. BASIC INTEGRATED ANALYSIS:")
        print("-" * 20)
        print(advanced_result.get('Basic Analysis', 'N/A'))
        
        print("\n6. QUANTITATIVE METRICS:")
        print("-" * 20)
        print(f"Founder Segmentation: {advanced_result.get('Founder Segmentation', 'N/A')}")
        print(f"Founder Idea Fit: {advanced_result.get('Founder Idea Fit', 'N/A')}")
        print(f"Categorical Prediction: {advanced_result.get('Categorical Prediction', 'N/A')}")
        print(f"Categorization: {advanced_result.get('Categorization', 'N/A')}")
        print(f"Quantitative Decision: {advanced_result.get('Quantitative Decision', 'N/A')}")
        print(f"Startup Info Parsed: {advanced_result.get('Startup Info', 'N/A')}")

    except Exception as e:
        print(f"\nError during advanced analysis: {str(e)}")
        import traceback
        traceback.print_exc()


    print("\n\n=== Testing Natural Language Analysis (analyze_startup_natural) ===")
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
        print(natural_result.get('Market Analysis', 'N/A')) # Use .get for safety
        
        print("\n2. PRODUCT ANALYSIS:")
        print("-" * 20)
        print(natural_result.get('Product Analysis', 'N/A')) # Use .get for safety
        
        print("\n3. FOUNDER ANALYSIS:")
        print("-" * 20)
        print(natural_result.get('Founder Analysis', 'N/A')) # Use .get for safety
        
        print("\n4. FINAL INTEGRATED ANALYSIS (PRO):")
        print("-" * 20)
        print(natural_result.get('Final Analysis', 'N/A')) # Use .get for safety

        print("\n5. BASIC INTEGRATED ANALYSIS:")
        print("-" * 20)
        print(natural_result.get('Basic Analysis', 'N/A')) # Use .get for safety
        
        print("\n6. QUANTITATIVE METRICS:")
        print("-" * 20)
        print(f"Founder Segmentation: {natural_result.get('Founder Segmentation', 'N/A')}")
        print(f"Founder Idea Fit: {natural_result.get('Founder Idea Fit', 'N/A')}")
        print(f"Categorical Prediction: {natural_result.get('Categorical Prediction', 'N/A')}")
        print(f"Categorization: {natural_result.get('Categorization', 'N/A')}")
        print(f"Quantitative Decision: {natural_result.get('Quantitative Decision', 'N/A')}")
        print(f"Startup Info Parsed: {natural_result.get('Startup Info', 'N/A')}")
        
    except Exception as e:
        print(f"\nError during natural language analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

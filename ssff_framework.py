import logging
from agents.market_agent import MarketAgent
from agents.product_agent import ProductAgent
from agents.founder_agent import FounderAgent
from agents.vc_scout_agent import VCScoutAgent
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

    def analyze_startup(self, startup_info, founder_description, mode="basic"):
        logger.info(f"Starting startup analysis in {mode} mode")

        # Get prediction
        prediction = self.vc_scout_agent.side_evaluate(startup_info)
        logger.info(f"VCScout prediction: {prediction}")

        if mode == "combined_pro":
            # Perform advanced computations
            founder_segmentation = self.founder_agent.segment_founder(founder_description)
            founder_idea_fit = self.founder_agent.calculate_idea_fit(startup_info, founder_description)
            rf_prediction = self.vc_scout_agent.get_rf_prediction(startup_info)

            # Enhance startup_info with advanced computations
            enhanced_startup_info = {
                **startup_info,
                "founder_segmentation": founder_segmentation,
                "founder_idea_fit": founder_idea_fit,
                "rf_prediction": rf_prediction
            }

            # Perform agent analyses with enhanced info
            market_analysis = self.market_agent.analyze(enhanced_startup_info, mode="advanced")
            product_analysis = self.product_agent.analyze(enhanced_startup_info, mode="advanced")
            founder_analysis = self.founder_agent.analyze(enhanced_startup_info, mode="advanced")

            # Add additional context to founder_analysis
            founder_analysis += (
                f"After modelling, the segmentation of the founder is {founder_segmentation}, "
                f"with L1 being least likely to be successful and L5 being most likely to be successful. "
                f"L5 founders are 3.8 times more likely to succeed than L1 founders. "
                f"The Founder_Idea_Fit Score of this startup is measured to be {founder_idea_fit}. "
                f"The score ranges from -1 to 1, with 1 being that the startup fits with the founder's "
                f"background well, and -1 being the least fit."
            )

            # Integrate analyses
            integrated_analysis = self.integration_agent.integrate_analyses(
                market_analysis, product_analysis, founder_analysis, prediction, mode
            )

            quant_decision = self.integration_agent.getquantDecision(
                prediction,
                founder_idea_fit,
                founder_segmentation,
                integrated_analysis
            )

            return {
                'Final Decision': integrated_analysis,
                'Market Info': market_analysis,
                'Product Info': product_analysis,
                'Founder Info': founder_analysis,
                'Market Report': self.market_agent.get_market_report(),
                'News Report': self.market_agent.get_news_report(),
                'Founder Segmentation': founder_segmentation,
                'Founder Idea Fit': founder_idea_fit,
                'Categorical Prediction': prediction,
                'Quantitative Decision': quant_decision,
                'Random Forest Prediction': rf_prediction
            }
        else:
            # Perform regular agent analyses
            market_analysis = self.market_agent.analyze(startup_info, mode)
            product_analysis = self.product_agent.analyze(startup_info, mode)
            founder_analysis = self.founder_agent.analyze(startup_info, mode)

            # Integrate analyses
            integrated_analysis = self.integration_agent.integrate_analyses(
                market_analysis, product_analysis, founder_analysis, prediction, mode
            )

            if mode == "basic":
                return {
                    'Final Decision': integrated_analysis,
                    'Market Info': market_analysis,
                    'Product Info': product_analysis,
                    'Founder Info': founder_analysis,
                    'Prediction': prediction
                }
            elif mode == "advanced":
                founder_segmentation = self.founder_agent.segment_founder(founder_description)
                founder_idea_fit = self.founder_agent.calculate_idea_fit(startup_info, founder_description)

                quant_decision = self.integration_agent.getquantDecision(
                    prediction,
                    founder_idea_fit,
                    founder_segmentation,
                    integrated_analysis
                )

                return {
                    'Final Decision': integrated_analysis,
                    'Market Info': market_analysis,
                    'Product Info': product_analysis,
                    'Founder Info': founder_analysis,
                    'Founder Segmentation': founder_segmentation,
                    'Founder Idea Fit': founder_idea_fit,
                    'Categorical Prediction': prediction,
                    'Quantitative Decision': quant_decision
                }
            else:
                logger.error(f"Invalid mode: {mode}")
                return None

def main():
    framework = StartupFramework()
    
    startup_info = {
        "name": "HealthTech AI",
        "description": "AI-powered health monitoring wearable device",
        "market_size": "$50 billion global wearable technology market",
        "competition": "Fitbit, Apple Watch, Garmin",
        "growth_rate": "CAGR of 15.9% from 2020 to 2027",
        "market_trends": "Increasing health consciousness, integration of AI in healthcare",
        "product_description": "Real-time health tracking with predictive analysis",
        "key_features": "Real-time health tracking, Personalized AI insights, Integration with medical systems",
        "tech_stack": "IoT sensors, Machine Learning algorithms, Cloud computing",
        "usp": "Predictive health analysis with medical-grade accuracy",
        "go_to_market_strategy": "B2C direct sales and partnerships with healthcare providers"
    }
    
    founder_description = "MBA from Stanford, 5 years at Google as Product Manager"

    basic_result = framework.analyze_startup(startup_info, founder_description, mode="basic")
    print("Basic Analysis Result:")
    print(basic_result)

    advanced_result = framework.analyze_startup(startup_info, founder_description, mode="advanced")
    print("\nAdvanced Analysis Result:")
    print(advanced_result)

    combined_pro_result = framework.analyze_startup(startup_info, founder_description, mode="combined_pro")
    print("\nCombined Pro Analysis Result:")
    print(combined_pro_result)

if __name__ == "__main__":
    main()
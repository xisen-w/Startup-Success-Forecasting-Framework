import os
import sys
import logging
from pydantic import BaseModel, Field

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from agents.base_agent import BaseAgent

class IntegratedAnalysis(BaseModel):
    overall_score: float = Field(..., description="Overall score between 1 and 10, 10 being the best")
    IntegratedAnalysis: str = Field(..., description="A comprehensive analysis of the startup from the perspective of the venture capital firm, after integrating all the information")
    recommendation: str = Field(..., description="A brief recommendation for next steps")
    outcome: str = Field(..., description="The outcome for prediction: 'Invest' or 'Hold' ")

class QuantitativeDecision(BaseModel):
    outcome: str = Field(..., description="Predicted outcome: 'Successful' or 'Unsuccessful'")
    probability: float = Field(..., description="Probability of the predicted outcome")
    reasoning: str = Field(..., description="One-line reasoning for the decision")

class IntegrationAgent(BaseAgent):
    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def integrated_analysis_basic(self, market_info, product_info, founder_info):
        self.logger.info("Starting basic integrated analysis")
        
        prompt = """
        Imagine you are the chief analyst at a venture capital firm, tasked with integrating the analyses of three specialized teams to provide a comprehensive investment insight. Your output should be structured with detailed scores and justifications:
        
        Example 1:
        Market Viability: 8.23/10 - The market is on the cusp of a regulatory shift that could open up new demand channels, supported by consumer trends favoring sustainability. Despite the overall growth, regulatory uncertainty poses a potential risk.
        Product Viability: 7.36/10 - The product introduces an innovative use of AI in renewable energy management, which is patent-pending. However, it faces competition from established players with deeper market penetration and brand recognition.
        Founder Competency: 9.1/10 - The founding team comprises industry veterans with prior successful exits and a strong network in the energy sector. Their track record includes scaling similar startups and navigating complex regulatory landscapes.
        
        Recommendation: Invest. The team's deep industry expertise and innovative product position it well to capitalize on the market's regulatory changes. Although competition is stiff, the founders' experience and network provide a competitive edge crucial for market adoption and navigating potential regulatory hurdles.
        
        Example 2:
        Market Viability: 5.31/10 - The market for wearable tech is saturated, with slow growth projections. However, there exists a niche but growing interest in wearables for pet health.
        Product Viability: 6.5/10 - The startup's product offers real-time health monitoring for pets, a feature not widely available in the current market. Yet, the product faces challenges with high production costs and consumer skepticism about the necessity of such a device.
        Founder Competency: 6.39/10 - The founding team includes passionate pet lovers with backgrounds in veterinary science and tech development. While they possess the technical skills and passion for the project, their lack of business and scaling experience is a concern.
        
        Recommendation: Hold. The unique product offering taps into an emerging market niche, presenting a potential opportunity. However, the combination of a saturated broader market, challenges in justifying the product's value to consumers, and the team's limited experience in business management suggests waiting for clearer signs of product-market fit and strategic direction.
        
        Now, analyze the following:
        
        Market Viability: {market_info}
        Product Viability: {product_info}
        Founder Competency: {founder_info}
        
        Provide an overall investment recommendation based on these inputs. State whether you would advise 'Invest' or 'Hold', including a comprehensive rationale for your decision.
        """
        
        user_prompt = prompt.format(
            market_info=market_info,
            product_info=product_info,
            founder_info=founder_info
        )
        
        integrated_analysis = self.get_json_response(IntegratedAnalysis, user_prompt, "Be professional.")
        self.logger.info("Basic integrated analysis completed")
        
        return integrated_analysis

    def integrated_analysis_pro(self, market_info, product_info, founder_info, founder_idea_fit, founder_segmentation, rf_prediction):
        self.logger.info("Starting pro integrated analysis")
        
        prompt = """
        Imagine you are the chief analyst at a venture capital firm, tasked with integrating the analyses of multiple specialized teams to provide a comprehensive investment insight. Your output should be structured with detailed scores and justifications:
        
        Example 1:
        Market Viability: 8.23/10 - The market is on the cusp of a regulatory shift that could open up new demand channels, supported by consumer trends favoring sustainability. Despite the overall growth, regulatory uncertainty poses a potential risk.
        Product Viability: 7.36/10 - The product introduces an innovative use of AI in renewable energy management, which is patent-pending. However, it faces competition from established players with deeper market penetration and brand recognition.
        Founder Competency: 9.1/10 - The founding team comprises industry veterans with prior successful exits and a strong network in the energy sector. Their track record includes scaling similar startups and navigating complex regulatory landscapes.
        
        Recommendation: Invest. The team's deep industry expertise and innovative product position it well to capitalize on the market's regulatory changes. Although competition is stiff, the founders' experience and network provide a competitive edge crucial for market adoption and navigating potential regulatory hurdles.
        
        Example 2:
        Market Viability: 5.31/10 - The market for wearable tech is saturated, with slow growth projections. However, there exists a niche but growing interest in wearables for pet health.
        Product Viability: 6.5/10 - The startup's product offers real-time health monitoring for pets, a feature not widely available in the current market. Yet, the product faces challenges with high production costs and consumer skepticism about the necessity of such a device.
        Founder Competency: 6.39/10 - The founding team includes passionate pet lovers with backgrounds in veterinary science and tech development. While they possess the technical skills and passion for the project, their lack of business and scaling experience is a concern.
        
        Recommendation: Hold. The unique product offering taps into an emerging market niche, presenting a potential opportunity. However, the combination of a saturated broader market, challenges in justifying the product's value to consumers, and the team's limited experience in business management suggests waiting for clearer signs of product-market fit and strategic direction.
        
        Now, analyze the following:
        
        Market Viability: {market_info}
        Product Viability: {product_info}
        Founder Competency: {founder_info}
        Founder-Idea Fit: {founder_idea_fit}
        Founder Segmentation: {founder_segmentation}
        Random Forest Prediction: {rf_prediction}

        Some context here for the scores: 
        1. Founder-Idea-Fit ranges from -1 to 1, a stronger number signifies a better fit.
        2. Founder Segmentation outcomes range from L1 to L5, with L5 being the most "competent" founders, and L1 otherwise.
        3. Random Forest Prediction predicts the expected outcome purely based on a statistical model, with an accuracy of around 65%.
        
        Provide an overall investment recommendation based on these inputs. State whether you would advise 'Invest' or 'Hold', including a comprehensive rationale for your decision. Consider all provided predictions and analyses, but do not over-rely on any single prediction.
        """
        
        user_prompt = prompt.format(
            market_info=market_info,
            product_info=product_info,
            founder_info=founder_info,
            founder_idea_fit=founder_idea_fit,
            founder_segmentation=founder_segmentation,
            rf_prediction=rf_prediction
        )
        
        integrated_analysis = self.get_json_response(IntegratedAnalysis, user_prompt, "Be professional.")
        self.logger.info("Pro integrated analysis completed")
        
        return integrated_analysis

    def getquantDecision(self, rf_prediction, Founder_Idea_Fit, Founder_Segmentation):
        self.logger.info("Starting quantitative decision analysis")
        
        prompt = """
        You are a final decision-maker. Think step by step. 
        
        You are now given Founder Segmentation. With L5 very likely to succeed and L1 least likely. You are also given the Founder-Idea Fit Score, with 1 being most fit and -1 being least fit. You are also given the result of prediction model (which should not be your main evidence because it may not be very accurate).
        
        This table summarises the implications of the Level Segmentation:
        
        Founder Level & Success & Failure & Success Rate & X-Time Better than L1 \\
        \midrule
        L1 & 24 & 75 & 24.24\% & 1 \\
        L2 & 83 & 223 & 27.12\% & 1.12 \\
        L3 & 287 & 445 & 39.21\% & 1.62 \\
        L4 & 514 & 249 & 67.37\% & 2.78 \\
        L5 & 93 & 8 & 92.08\% & 3.79 \\
        
        Regarding the Founder-Idea-Fit Score. Relevant context are provided here: 
        The previous sections show the strong correlation between founder's segmentation level and startup's outcome, as L5 founders are more than three times likely to succeed than L1 founders. However, looking into the data, one could also see that there are L5 founders who did not succeed, and there are L1 founders who succeeded. To account for these scenarios, we investigate the fit between founders and their ideas.
        
        To assess quantitatively, we propose a metric called Founder-Idea Fit Score (FIFS). The Founder-Idea Fit Score quantitatively assesses the compatibility between a founder's experience level and the success of their startup idea. Given the revised Preliminary Fit Score ($PFS$) defined as:
        \[PFS(F, O) = (6 - F) \times O - F \times (1 - O)\]
        where $F$ represents the founder's level ($1$ to $5$) and $O$ is the outcome ($1$ for success, $0$ for failure), we aim to normalize this score to a range of $[-1, 1]$ to facilitate interpretation.
        
        To achieve this, we note that the minimum $PFS$ value is $-5$ (for a level $5$ founder who fails), and the maximum value is $5$ (for a level $1$ founder who succeeds). The normalization formula to scale $PFS$ to $[-1, 1]$ is:
        \[Normalized\;PFS = \frac{PFS}{5}\]
        
        Now use all of these information, produce a string of the predicted outcome and probability, with one line of reasoning. 
        
        Your response should be in the following format:
        {
          "outcome": "<Successful or Unsuccessful>",
          "probability": <probability as a float between 0 and 1>,
          "reasoning": "<One-line reasoning for the decision>"
        }

        You will also receive a categorical prediction outcome of the prediction model (which should not be your main evidence because it may not be very accurate, just around 65% accuracy).
        
        Ensure that your response is a valid JSON object and includes all the fields mentioned above.
        """

        user_prompt = f"You are provided with the categorical prediction outcome of {rf_prediction}, Founder Segmentation of {Founder_Segmentation}, Founder-Idea Fit of {Founder_Idea_Fit}."

        quant_decision = self.get_json_response(QuantitativeDecision, prompt, user_prompt)
        self.logger.info("Quantitative decision analysis completed")
        
        return quant_decision

if __name__ == "__main__":
    def test_integration_agent():
        # Configure logging for the test function
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info("Starting IntegrationAgent test")
        # Create an IntegrationAgent instance
        agent = IntegrationAgent()

        # Test market info
        market_info = """
        Market size: $50 billion global wearable technology market
        Growth rate: CAGR of 15.9% from 2020 to 2027
        Competition: Fitbit, Apple Watch, Garmin
        Market trends: Increasing health consciousness, integration of AI in healthcare
        Viability score: 8
        """

        # Test product info
        product_info = """
        Features analysis: Real-time health tracking with predictive analysis
        Tech stack evaluation: IoT sensors, Machine Learning algorithms, Cloud computing
        USP assessment: Predictive health analysis with medical-grade accuracy
        Potential score: 9
        Innovation score: 8
        Market fit score: 7
        """

        # Test founder info
        founder_info = """
        Competency score: 8
        Strengths: Strong technical background, previous startup experience
        Challenges: Limited experience in the healthcare industry
        Segmentation: L4
        Idea fit: 0.85
        """

        # Test basic integration
        print("Basic Integration:")
        basic_integration = agent.integrated_analysis_basic(market_info, product_info, founder_info)
        print(basic_integration)
        print()

        # Test advanced integration
        print("Advanced Integration:")
        advanced_integration = agent.integrated_analysis_pro(
            market_info,
            product_info,
            founder_info,
            founder_idea_fit=0.85,
            founder_segmentation="L4",
            rf_prediction="Successful"
        )
        print(advanced_integration)

        # Test quantitative decision
        print("Quantitative Decision:")
        quant_decision = agent.getquantDecision(
            rf_prediction="Successful",
            Founder_Idea_Fit=0.85,
            Founder_Segmentation="L4"
        )
        print(quant_decision)

    # Run the test function
    test_integration_agent()

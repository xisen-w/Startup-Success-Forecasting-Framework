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
    summary: str = Field(..., description="A brief summary of the startup's potential")
    strengths: list[str] = Field(..., description="List of key strengths")
    weaknesses: list[str] = Field(..., description="List of potential weaknesses")
    recommendation: str = Field(..., description="A brief recommendation for next steps")
    outcome: str = Field(..., description="The outcome for prediction: 'Successful' or 'Unsuccessful' ")

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

    def integrate_analyses(self, market_info, product_info, founder_info, prediction, 
                           founder_idea_fit, founder_segmentation, rf_prediction, 
                           categorization, mode="basic"):
        self.logger.info(f"Starting integration of analyses in {mode} mode")
        
        combined_info = f"Market Analysis:\n{market_info}\n\n" \
                        f"Product Analysis:\n{product_info}\n\n" \
                        f"Founder Analysis:\n{founder_info}\n\n" \
                        f"Prediction: {prediction}\n" \
                        f"Founder-Idea Fit: {founder_idea_fit}\n" \
                        f"Founder Segmentation: {founder_segmentation}\n" \
                        f"Random Forest Prediction: {rf_prediction}\n" \
                        f"Categorization: {categorization}"
        
        self.logger.debug(f"Combined info: {combined_info}")
        
        integrated_analysis = self.get_json_response(IntegratedAnalysis, self._get_integration_prompt(mode), combined_info)
        self.logger.info("Integration completed")
        
        return integrated_analysis

    def getquantDecision(self, prediction, Founder_Idea_Fit, Founder_Segmentation, final_decision):
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
        
        In addition, you are given the scores of analysis on Market Viability, Product Viability, and Founder Competency in the form of a report. Ignore any analysis in the report but do focus on the numbers. [Important: Focus on the scores, IGNORE any other analysis of the final report. The rest of analysis is meaningless to you and should not affect any of your decision-making.] 
        
        Now use all of these information, produce a string of the predicted outcome and probability, with one line of reasoning. 
        
        Your response should be in the following format:
        {
          "outcome": "<Successful or Unsuccessful>",
          "probability": <probability as a float between 0 and 1>,
          "reasoning": "<One-line reasoning for the decision>"
        }
        
        Ensure that your response is a valid JSON object and includes all the fields mentioned above.
        """

        user_prompt = f"You are provided with the categorical prediction outcome of {prediction}, Founder Segmentation of {Founder_Segmentation}, Founder-Idea Fit of {Founder_Idea_Fit}. Finally, here's the report that contains the score breakdown:{final_decision}."

        quant_decision = self.get_json_response(QuantitativeDecision, prompt, user_prompt)
        self.logger.info("Quantitative decision analysis completed")
        
        return quant_decision

    def _get_integration_prompt(self, mode):
        base_prompt = """
        As the chief analyst at a venture capital firm, integrate the following analyses into a cohesive evaluation:
        {combined_info}

        Synthesize the information and provide an overall assessment of the startup's potential.
        Consider all provided predictions and analyses, including the Founder-Idea Fit, Founder Segmentation, 
        Random Forest Prediction, and Categorization.
        Score each aspect from 1 to 10 (10 is the best & most competitive). Specify the score to 2 digits and give very strong justification for it.

        Your response should be structured as follows:
        1. Overall Score: A number between 1 and 10, calculated as a weighted average of the individual scores.
        2. Summary: A brief summary of the startup's potential, highlighting key points from each analysis and prediction.
        3. Strengths: A list of key strengths identified across all analyses.
        4. Weaknesses: A list of potential weaknesses or challenges identified.
        5. Recommendation: A brief recommendation for next steps, whether to invest or not, and any conditions or areas for further investigation.
        6. Prediction: A final decision: successful or unsuccessful, considering all provided predictions and analyses.
        
        Ensure that your response is a valid JSON object and includes all the fields mentioned above.
        """

        if mode == "advanced":
            base_prompt += """
            Additionally, provide a brief explanation of how the various predictions (LLM, Random Forest, Founder-Idea Fit, 
            Founder Segmentation) influenced your final assessment and recommendation.
            """

        return base_prompt

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
        basic_integration = agent.integrate_analyses(market_info, product_info, founder_info, mode="basic")
        print(basic_integration)
        print()

        # Test advanced integration
        print("Advanced Integration:")
        advanced_integration = agent.integrate_analyses(market_info, product_info, founder_info, prediction="Successful", mode="advanced")
        print(advanced_integration)

        # Test quantitative decision
        print("Quantitative Decision:")
        quant_decision = agent.getquantDecision(
            prediction="Successful",
            Founder_Idea_Fit=0.85,
            Founder_Segmentation="L4",
            final_decision="Market Viability: 8, Product Viability: 9, Founder Competency: 8"
        )
        print(quant_decision)

    # Run the test function
    test_integration_agent()

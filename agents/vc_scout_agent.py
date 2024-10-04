import os
import sys
import logging
import joblib
import pandas as pd
from pydantic import BaseModel, Field

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, project_root)

from agents.base_agent import BaseAgent

class StartupEvaluation(BaseModel):
    market_opportunity: str = Field(..., description="Assessment of the market opportunity")
    product_innovation: str = Field(..., description="Evaluation of product innovation")
    founding_team: str = Field(..., description="Analysis of the founding team")
    potential_risks: str = Field(..., description="Identification of potential risks")
    overall_potential: int = Field(..., description="Overall potential score on a scale of 1 to 10")
    investment_recommendation: str = Field(None, description="Investment recommendation: 'Invest' or 'Pass'")
    confidence: float = Field(None, description="Confidence level in the recommendation (0 to 1)")
    rationale: str = Field(None, description="Brief explanation for the recommendation")

class StartupAnalysisResponses(BaseModel):
    industry_growth: str = Field(..., description="Is the startup operating in an industry experiencing growth?")
    market_size: str = Field(..., description="Is the target market size for the startup's product/service considered large?")
    development_pace: str = Field(..., description="Does the startup demonstrate a fast pace of development compared to competitors?")
    market_adaptability: str = Field(..., description="Is the startup considered adaptable to market changes?")
    execution_capabilities: str = Field(..., description="How would you rate the startup's execution capabilities?")
    funding_amount: str = Field(..., description="Has the startup raised a significant amount of funding in its latest round?")
    valuation_change: str = Field(..., description="Has the startup's valuation increased with time?")
    investor_backing: str = Field(..., description="Are well-known investors or venture capital firms backing the startup?")
    reviews_testimonials: str = Field(..., description="Are the reviews and testimonials for the startup predominantly positive?")
    product_market_fit: str = Field(..., description="Do market surveys indicate a strong product-market fit for the startup?")
    sentiment_analysis: str = Field(..., description="Does the sentiment analysis of founder and company descriptions suggest high positivity?")
    innovation_mentions: str = Field(..., description="Are terms related to innovation frequently mentioned in the company's public communications?")
    cutting_edge_technology: str = Field(..., description="Does the startup mention cutting-edge technology in its descriptions?")
    timing: str = Field(..., description="Considering the startup's industry and current market conditions, is the timing for the startup's product or service right?")

class VCScoutAgent(BaseAgent):
    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Load the encoder and model
        self.encoder = joblib.load(os.path.join(project_root, 'models/trained_encoder_RF.joblib'))
        self.model_random_forest = joblib.load(os.path.join(project_root,'models/random_forest_classifier.joblib'))

    def evaluate(self, startup_info, mode):
        self.logger.info(f"Starting startup evaluation in {mode} mode")
        startup_info_str = self._get_startup_info(startup_info)
        self.logger.debug(f"Startup info: {startup_info_str}")
        
        if mode == "basic":
            analysis = self.get_json_response(StartupEvaluation, self._get_basic_evaluation_prompt(), startup_info_str)
            self.logger.info("Basic evaluation completed")
        else:  # advanced mode
            analysis = self.get_json_response(StartupEvaluation, self._get_advanced_evaluation_prompt(), startup_info_str)
            self.logger.info("Advanced evaluation completed")
        
        return analysis

    def side_evaluate(self, startup_info):
        self.logger.info("Starting side evaluation")
        system_content = """
        As an analyst specializing in startup evaluation, your task is to categorize startups based on specific criteria related to their market, financial performance, product, team, funding, customer feedback, operational efficiency, and technological innovation. For each of the following questions, please provide a categorical response: 'Yes', 'No', 'N/A', 'Small', 'Medium', 'Large', 'Slower', 'Same', 'Faster', 'Not Adaptable', 'Somewhat Adaptable', 'Very Adaptable', 'Poor', 'Average', 'Excellent', 'Below Average', 'Above Average', 'Decreased', 'Remained Stable', 'Increased', 'Unknown', 'Recognized', 'Highly Regarded', 'Negative', 'Mixed', 'Positive', 'Weak', 'Moderate', 'Strong', 'Less Efficient', 'Efficient', 'More Efficient', 'Higher', 'Same', 'Lower', 'Fragile', 'Adequate', 'Robust', 'Rarely', 'Sometimes', 'Often', 'No', 'Mentioned', 'Emphasized', 'Too Early', 'Just Right', 'Too Late'.

        Think step by step and analyze with reasoning. Think critically and analyze carefully. But do not generate anything other than the JSON itself. 

        Provide your analysis in the following JSON format:
        {
          "industry_growth": "<Your response>",
          "market_size": "<Your response>",
          "development_pace": "<Your response>",
          "market_adaptability": "<Your response>",
          "execution_capabilities": "<Your response>",
          "funding_amount": "<Your response>",
          "valuation_change": "<Your response>",
          "investor_backing": "<Your response>",
          "reviews_testimonials": "<Your response>",
          "product_market_fit": "<Your response>",
          "sentiment_analysis": "<Your response>",
          "innovation_mentions": "<Your response>",
          "cutting_edge_technology": "<Your response>",
          "timing": "<Your response>"
        }

        Ensure that your response is a valid JSON object and includes all the fields mentioned above.
        """
        
        categories = self.get_json_response(StartupAnalysisResponses, system_content, startup_info)
        self.logger.info("Categories & Result: %s", categories)

        # The adjusted category mappings with 'Mismatch' included
        category_mappings = {
            "industry_growth": ["No", "N/A", "Yes", "Mismatch"],
            "market_size": ["Small", "Medium", "Large", "N/A", "Mismatch"],
            "development_pace": ["Slower", "Same", "Faster", "N/A", "Mismatch"],
            "market_adaptability": ["Not Adaptable", "Somewhat Adaptable", "Very Adaptable", "N/A", "Mismatch"],
            "execution_capabilities": ["Poor", "Average", "Excellent", "N/A", "Mismatch"],
            "funding_amount": ["Below Average", "Average", "Above Average", "N/A", "Mismatch"],
            "valuation_change": ["Decreased", "Remained Stable", "Increased", "N/A", "Mismatch"],
            "investor_backing": ["Unknown", "Recognized", "Highly Regarded", "N/A", "Mismatch"],
            "reviews_testimonials": ["Negative", "Mixed", "Positive", "N/A", "Mismatch"],
            "product_market_fit": ["Weak", "Moderate", "Strong", "N/A", "Mismatch"],
            "sentiment_analysis": ["Negative", "Neutral", "Positive", "N/A", "Mismatch"],
            "innovation_mentions": ["Rarely", "Sometimes", "Often", "N/A", "Mismatch"],
            "cutting_edge_technology": ["No", "Mentioned", "Emphasized", "N/A", "Mismatch"],
            "timing": ["Too Early", "Just Right", "Too Late", "N/A", "Mismatch"]
        }

        # The order of features as used during training
        feature_order = list(category_mappings.keys())

        # Call the function with our special encoder & trained model
        prediction = self.preprocess_and_predict(categories.dict(), category_mappings, self.encoder, self.model_random_forest, feature_order)

        if prediction[0] == 1:
            prediction = "Successful"
        else:
            prediction = "Unsuccessful"

        self.logger.info("Prediction: %s", prediction)
        return prediction

    def preprocess_and_predict(self, single_instance, category_mappings, encoder_special, model, feature_order):
        # Convert single instance dictionary into DataFrame
        single_instance_df = pd.DataFrame([single_instance])

        # Preprocess single_instance_df to match training feature names and order
        for column in feature_order:
            if column not in single_instance_df:
                single_instance_df[column] = "Mismatch"  # Add missing columns as "Mismatch"

        # Ensure DataFrame columns are in the same order as during training
        single_instance_df = single_instance_df[feature_order]

        # Replace categories not in mappings with "Mismatch"
        for column, categories in category_mappings.items():
            single_instance_df[column] = single_instance_df[column].apply(lambda x: x if x in categories else "Mismatch")

        self.logger.debug("Encoder categories: %s", encoder_special.categories_)
        # Encode the single instance using the trained OrdinalEncoder
        single_instance_encoded = encoder_special.transform(single_instance_df)

        # Use the trained model to predict
        prediction = model.predict(single_instance_encoded)

        return prediction

    def _get_startup_info(self, startup_info):
        return f"Startup Name: {startup_info.get('name', '')}\n" \
               f"Description: {startup_info.get('description', '')}\n" \
               f"Market Size: {startup_info.get('market_size', '')}\n" \
               f"Competition: {startup_info.get('competition', '')}\n" \
               f"Growth Rate: {startup_info.get('growth_rate', '')}\n" \
               f"Market Trends: {startup_info.get('market_trends', '')}\n" \
               f"Product Description: {startup_info.get('product_description', '')}\n" \
               f"Founding Team: {startup_info.get('founding_team', '')}\n" \
               f"Funding Stage: {startup_info.get('funding_stage', '')}\n" \
               f"Funding Amount: {startup_info.get('funding_amount', '')}"

    def _get_basic_evaluation_prompt(self):
        return """
        As an experienced VC scout, evaluate the overall potential of this startup based on the following information:
        {startup_info}

        Provide your evaluation in the following JSON format:
        {
            "market_opportunity": "Your detailed assessment of the market opportunity",
            "product_innovation": "Your comprehensive evaluation of the product innovation",
            "founding_team": "Your thorough analysis of the founding team",
            "potential_risks": "Your identification of potential risks",
            "overall_potential": <integer between 1 and 10 representing the overall potential>
        }

        Ensure that your response is a valid JSON object and includes all the fields mentioned above.
        """

    def _get_advanced_evaluation_prompt(self):
        return """
        As an experienced VC scout, provide a comprehensive evaluation of this startup and an investment recommendation based on the following information:
        {startup_info}

        Provide your evaluation and recommendation in the following JSON format:
        {
            "market_opportunity": "Your detailed assessment of the market opportunity",
            "product_innovation": "Your comprehensive evaluation of the product innovation",
            "founding_team": "Your thorough analysis of the founding team",
            "potential_risks": "Your identification of potential risks",
            "overall_potential": <integer between 1 and 10 representing the overall potential>,
            "investment_recommendation": "<Either 'Invest' or 'Pass'>",
            "confidence": <float between 0 and 1 representing your confidence in the recommendation>,
            "rationale": "Your brief explanation for the investment recommendation"
        }

        Ensure that your response is a valid JSON object and includes all the fields mentioned above.
        """

if __name__ == "__main__":
    def test_vc_scout_agent():
        # Configure logging for the test function
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info("Starting VCScoutAgent test")
        # Create a VCScoutAgent instance
        agent = VCScoutAgent()

        # Test startup info
        startup_info = {
            "name": "HealthTech AI",
            "description": "AI-powered health monitoring wearable device",
            "market_size": "$50 billion global wearable technology market",
            "competition": "Fitbit, Apple Watch, Garmin",
            "growth_rate": "CAGR of 15.9% from 2020 to 2027",
            "market_trends": "Increasing health consciousness, integration of AI in healthcare",
            "product_description": "Real-time health tracking with predictive analysis",
            "founding_team": "Experienced entrepreneurs with backgrounds in AI and healthcare",
            "funding_stage": "Seed",
            "funding_amount": "$2 million raised to date"
        }

        startup_info = agent._get_startup_info(startup_info)

        # Test basic evaluation
        print("Basic Evaluation:")
        basic_evaluation = agent.evaluate(startup_info, mode="basic")
        print(basic_evaluation)
        print()

        # Test advanced evaluation
        print("Advanced Evaluation:")
        advanced_evaluation = agent.evaluate(startup_info, mode="advanced")
        print(advanced_evaluation)
        print()

        # Test side evaluation
        print("Side Evaluation:")
        side_evaluation = agent.side_evaluate(startup_info)
        print(side_evaluation)

    # Run the test function
    test_vc_scout_agent()
from agents.base_agent import BaseAgent

class VCScoutAgent(BaseAgent):
    def __init__(self, model=Config.DEFAULT_MODEL):
        super().__init__(model)

    def evaluate(self, startup_info, mode):
        evaluation = self.get_response(self._get_evaluation_prompt(), startup_info)
        
        if mode == "advanced":
            investment_recommendation = self._get_investment_recommendation(startup_info)
            evaluation += f"\nInvestment Recommendation: {investment_recommendation}"
        
        return evaluation

    def _get_investment_recommendation(self, startup_info):
        return self.get_json_response(self._get_recommendation_prompt(), startup_info)

    def _get_evaluation_prompt(self):
        return """
        As an experienced VC scout, evaluate the overall potential of this startup based on the following information:
        {startup_info}

        Consider the market opportunity, product innovation, founding team, and potential risks.
        Provide a comprehensive evaluation and rate the startup's overall potential on a scale of 1 to 10.
        """

    def _get_recommendation_prompt(self):
        return """
        Based on the following startup information, provide an investment recommendation:
        {startup_info}

        Return your response as a JSON object with the following structure:
        {
            "recommendation": "Invest" or "Pass",
            "confidence": A number between 0 and 1,
            "rationale": "A brief explanation of your recommendation"
        }
        """
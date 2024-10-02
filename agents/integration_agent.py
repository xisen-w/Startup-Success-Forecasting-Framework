from agents.base_agent import BaseAgent

class IntegrationAgent(BaseAgent):
    def __init__(self, model=Config.DEFAULT_MODEL):
        super().__init__(model)

    def integrate_analyses(self, analyses):
        integrated_analysis = self.get_json_response(self._get_integration_prompt(), str(analyses))
        return integrated_analysis

    def _get_integration_prompt(self):
        return """
        As an expert startup analyst, integrate the following analyses into a cohesive evaluation:
        {analyses}

        Synthesize the information and provide an overall assessment of the startup's potential.
        Return your response as a JSON object with the following structure:
        {
            "overall_score": A number between 1 and 10,
            "summary": "A brief summary of the startup's potential",
            "strengths": ["List", "of", "key", "strengths"],
            "weaknesses": ["List", "of", "potential", "weaknesses"],
            "recommendation": "A brief recommendation for next steps"
        }
        """
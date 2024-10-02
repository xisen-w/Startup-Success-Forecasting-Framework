from agents.base_agent import BaseAgent

class ProductAgent(BaseAgent):
    def __init__(self, model=Config.DEFAULT_MODEL):
        super().__init__(model)

    def analyze(self, startup_info, mode):
        product_info = self._get_product_info(startup_info)
        analysis = self.get_response(self._get_analysis_prompt(), product_info)
        
        if mode == "advanced":
            innovation_score = self._assess_innovation(product_info)
            market_fit = self._assess_market_fit(startup_info, product_info)
            analysis += f"\nInnovation Score: {innovation_score}\nMarket Fit: {market_fit}"
        
        return analysis

    def _get_product_info(self, startup_info):
        return f"Product Description: {startup_info.get('product_description', '')}\n" \
               f"Key Features: {startup_info.get('key_features', '')}\n" \
               f"Technology Stack: {startup_info.get('tech_stack', '')}\n" \
               f"Unique Selling Proposition: {startup_info.get('usp', '')}"

    def _assess_innovation(self, product_info):
        return self.get_response(self._get_innovation_prompt(), product_info)

    def _assess_market_fit(self, startup_info, product_info):
        combined_info = f"{startup_info.get('market_description', '')}\n{product_info}"
        return self.get_response(self._get_market_fit_prompt(), combined_info)

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
        Provide a brief explanation for your rating.
        """

    def _get_market_fit_prompt(self):
        return """
        Evaluate the product-market fit based on the following information about the market and the product:
        {combined_info}

        Rate the product-market fit on a scale of 1 to 10, where 1 is poor fit and 10 is perfect fit.
        Provide a brief explanation for your rating.
        """
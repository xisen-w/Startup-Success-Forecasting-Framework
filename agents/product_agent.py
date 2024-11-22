import os
import sys
import logging

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from agents.base_agent import BaseAgent
from utils.api_wrapper import GoogleSearchAPI
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
        self.search_api = GoogleSearchAPI()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Add file handler for external knowledge logging
        self.external_knowledge_logger = logging.getLogger(f"{__name__}.external_knowledge")
        file_handler = logging.FileHandler('experiments/product_external_knowledge.log')
        file_handler.setFormatter(formatter)
        self.external_knowledge_logger.addHandler(file_handler)
        self.external_knowledge_logger.setLevel(logging.INFO)

    def analyze(self, startup_info, mode):
        self.logger.info(f"Starting product analysis in {mode} mode")
        product_info = self._get_product_info(startup_info)
        
        if mode == "natural_language_advanced":
            self.logger.info("Starting advanced analysis")
            keywords = self._generate_keywords(startup_info)
            
            # Get the structured product report
            product_report = self._get_external_knowledge(startup_info)
            self.logger.debug(f"Product report: {product_report}")
            
            prompt = self.natural_language_analysis_prompt().format(
                startup_info=startup_info,
                product_info=product_info,
                keywords=keywords,
                external_knowledge=product_report
            )
            
            nl_advanced_analysis = self.get_response(prompt, "Write a comprehensive report about the product analysis from the VC perspective.")
            self.logger.info("Natural language analysis completed")
            return nl_advanced_analysis
        
        if mode == "advanced":
            self.logger.info("Starting advanced analysis with external research")
            external_knowledge = self._get_external_knowledge(startup_info)
            
            # Log external knowledge details
            self.external_knowledge_logger.info("\n" + "="*50)
            self.external_knowledge_logger.info(f"External Knowledge for: {startup_info.get('name', 'Unnamed Startup')}")
            self.external_knowledge_logger.info("="*50)
            self.external_knowledge_logger.info(f"\nProduct Info:\n{product_info}")
            self.external_knowledge_logger.info(f"\nExternal Knowledge:\n{external_knowledge}")
            self.external_knowledge_logger.info("="*50 + "\n")
            
            analysis = self.get_json_response(
                ProductAnalysis, 
                self._get_advanced_analysis_prompt(), 
                f"{product_info}\n\nExternal Research:\n{external_knowledge}"
            )
            self.logger.info("Advanced analysis completed")
            return analysis

            
        analysis = self.get_json_response(ProductAnalysis, self._get_analysis_prompt(), product_info)
        self.logger.info("Basic analysis completed")
        return analysis
    

    def _get_external_knowledge(self, startup_info):
        """Get structured product research from external sources"""
        self.logger.info("Starting external knowledge gathering")
        
        # Generate keyword (simpler approach matching the pipeline)
        keywords = startup_info.get('name', '')
        keywords += " News"
        self.logger.info(f"Generated keywords: {keywords}")
        
        # Get search results
        search_results = self.search_api.search(keywords)[:20]
        self.logger.info("Raw search results received")
        print("\nRAW SEARCH RESULTS:")
        print("-" * 40)
        print(search_results)
        
        # Process organic results
        organic_knowledge = ""
        if isinstance(search_results, list):  # Direct list of results
            organic_results = search_results[:10]
            for result in organic_results:
                # Extract the most valuable fields
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                source = result.get('source', '')
                
                # Include sitelinks if they exist (they often contain valuable additional info)
                sitelinks = result.get('sitelinks', {}).get('expanded', [])
                sitelinks_info = ""
                if sitelinks:
                    sitelinks_info = "\nRelated pages:\n" + "\n".join(
                        [f"- {link['title']}: {link['snippet']}" 
                         for link in sitelinks if 'snippet' in link][:3]
                    )
                
                organic_knowledge += f"\nSource: {source}\nTitle: {title}\nSummary: {snippet}{sitelinks_info}\n"
            
            print("\nPROCESSED ORGANIC RESULTS:")
            print("-" * 40)
            print(organic_knowledge)
        
        # Process related questions (only if search_results is a dict)
        brainstorm_results = ""
        if isinstance(search_results, dict):
            related_QAs = search_results.get("related_questions", [])
            if related_QAs:
                for qa in related_QAs:
                    title = qa.get('title', "")
                    question = qa.get('question', "")
                    snippet = qa.get('snippet', "")
                    date = qa.get('date', "")
                    brainstorm_results += f"Title: {title} + Question: {question} + Snippet: {snippet} + Date: {date}\n"
                print("\nPROCESSED RELATED QUESTIONS:")
                print("-" * 40)
                print(brainstorm_results)
        
        # Process related news (only if search_results is a dict)
        related_news = " "
        if isinstance(search_results, dict):
            top_stories = search_results.get("top_stories", [])
            if top_stories:
                for story in top_stories:
                    title = story.get('title', "")
                    source = story.get('source', "")
                    date = story.get('date', "")
                    related_news += f"Title: {title} + Source: {source} + Date:{date} + \n"
                print("\nPROCESSED RELATED NEWS:")
                print("-" * 40)
                print(related_news)
        
        # Compile overall knowledge
        overall_knowledge = (
            "Here are the web search information about the company:\n" + organic_knowledge +
            "\nHere are the contexts:\n" + brainstorm_results +
            "\nHere are the related news:\n" + related_news
        )
        
        print("\nSTRUCTURED KNOWLEDGE:")
        print("-" * 40)
        print(overall_knowledge)
        
        synthesis_prompt = ("You will assist me in summarising the latest information and news about the company. "
                           "After google search, you are given important context information and data (most of the time). "
                           "Now please summarise the information as a report to highlight the latest information and "
                           "public sentiment towards the company and its product, alongside with your existing knowledge. "
                           "Make your response structured and in detail.")
        
        product_report = self.get_response(synthesis_prompt, overall_knowledge)
        print("\nFINAL PRODUCT REPORT:")
        print("-" * 40)
        print(product_report)
        
        return product_report

    def _generate_keywords(self, startup_info):
        company_name = startup_info.get('name', '')
        prompt = f"""Generate 3-5 specific search keywords about {company_name}, starting with the company name itself.
        
        Company: {company_name}
        Product: {startup_info.get('product_description', '')}
        Features: {startup_info.get('key_features', '')}
        Technology: {startup_info.get('tech_stack', '')}
        
        Guidelines for keywords:
        1. FIRST keyword should be just the company name: "{company_name}"
        2. Other keywords should include company name with product type
        3. Include specific product features or competitor comparison
        
        Example for Apple Inc:
        - "Apple Inc"
        - "Apple Watch health features"
        - "Apple Watch vs Fitbit comparison"
        
        Example for Tesla:
        - "Tesla"
        - "Tesla Autopilot capabilities"
        - "Tesla vs Waymo self-driving"
        
        Generate specific keywords for {company_name}:
        """
        
        # Log keyword generation prompt
        self.external_knowledge_logger.info(f"\nKeyword Generation Prompt:\n{prompt}")
        
        keywords = self.get_response(self._get_keyword_generation_prompt(), prompt)
        return keywords

    def _get_product_info(self, startup_info):
        return f"Product Description: {startup_info.get('product_details', '')}\n" \
               f"Key Features: {startup_info.get('product_details', '')}\n" \
               f"Technology Stack: {startup_info.get('technology_stack', '')}\n" \
               f"Unique Selling Proposition: {startup_info.get('product_fit', '')}"

    def _get_analysis_prompt(self):
        return """
        As a product expert, analyze the startup's product based on the following information:
        {product_info}

        Consider the product's features, technology stack, and unique selling proposition.
        Provide a comprehensive analysis and rate the product's potential on a scale of 1 to 10.

        Make sure that you think step by step and analyze in a professional manner.
        """

    def _get_advanced_analysis_prompt(self):
        return """
        As a product expert, provide an in-depth analysis of the startup's product based on the following information:
        {product_info}

        Include insights from external research about similar products, technical feasibility, and industry standards.
        
        Consider:
        1. Technical innovation and feasibility
        2. Feature completeness compared to competitors
        3. Technology stack robustness
        4. Unique selling proposition strength
        5. Implementation challenges and solutions
        
        Provide scores for:
        - Product potential (1-10)
        - Innovation level (1-10)
        - Market fit (1-10)
        
        Make sure to think step by step and analyze in a professional manner.
        """

    def _get_keyword_generation_prompt(self):
        return """You are an AI assistant skilled at generating relevant search keywords 
        for technical and product research. Please provide 3-5 concise keywords or short 
        phrases based on the given information, focusing on technical aspects and industry standards."""

    def _get_synthesis_prompt(self):
        return """You are an AI assistant skilled at synthesizing technical information. 
        Please provide a concise summary of the key points from the given search results, 
        focusing on technical feasibility, similar products, and industry standards."""

    def natural_language_analysis_prompt(self):
        return """
        You are a professional product analyst in a VC firm evaluating a potential investment opportunity. 
        
        Company Information:
        {startup_info}

        Product Information:
        {product_info}

        Product Research Report:
        {external_knowledge}

        Based on this comprehensive product research and initial data, please provide:
        1. Technical Innovation Analysis
           - How innovative is the technology?
           - Is it feasible to implement?
           - What are the technical risks?

        2. Feature Set Evaluation
           - How complete is the feature set?
           - How does it compare to competitors?
           - What are the key differentiators?

        3. Implementation Assessment
           - What are the main technical challenges?
           - How realistic is the development timeline?
           - What resources are needed?

        4. Market Readiness
           - Is the product ready for the target market?
           - What further development is needed?
           - How strong is the product-market fit?

        Please reference specific data points from the product research report in your analysis.
        Conclude with:
        - Product potential score (1-10)
        - Innovation score (1-10)
        - Market fit score (1-10)
        """

if __name__ == "__main__":
    def test_product_agent():
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        logger.info("Starting ProductAgent test")
        
        # Create agent instance
        agent = ProductAgent()

        # Test startup info (Stripe example)
        startup_info = {
            "name": "Stripe",
            "description": "Developer-first payment processing platform that allows businesses to accept and manage online payments through simple API integration.",
            "product_details": """
                - RESTful API for payment processing
                - Support for 135+ currencies
                - Fraud prevention system
                - Real-time reporting dashboard
                - Subscription management
                - Invoice automation
                - Mobile SDK
            """,
            "technology_stack": """
                - Ruby on Rails backend
                - React.js frontend
                - PostgreSQL database
                - Redis for caching
                - AWS infrastructure
                - Machine learning for fraud detection
                - Kubernetes for orchestration
            """,
            "product_fit": "Simplified payments integration for developers with robust API documentation and extensive feature set"
        }

        print("\n=== Starting Analysis of Stripe's Product ===")
        print("-" * 80)
        
        # The _get_external_knowledge function already handles and logs:
        # - keyword generation
        # - raw search results
        # - structured knowledge
        # - final synthesis
        external_knowledge = agent._get_external_knowledge(startup_info)
        
        # Full analysis using the gathered knowledge
        print("\nFULL PRODUCT ANALYSIS:")
        print("-" * 40)
        analysis = agent.analyze(startup_info, mode="natural_language_advanced")
        print(analysis)

    # Run the test
    test_product_agent()
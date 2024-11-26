import os
import sys
import logging
from pydantic import BaseModel, Field
from utils.api_wrapper import get_structured_output

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from agents.base_agent import BaseAgent

class BaselineAnalysis(BaseModel):
    total_analysis: str = Field(..., description="Detailed analysis of the startup")
    score: float = Field(..., description="Overall score between 1 and 10")
    recommendation: str = Field(..., description="Recommendation: 'Successful' or 'Unsuccessful'")

class BaselineFramework(BaseAgent):
    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)

    def analyze_startup(self, startup_info_str: str) -> dict:
        """Simple baseline analysis using only ChatGPT"""
        self.logger.info("Starting baseline analysis")
        
        prompt = """
        You are an experienced venture capitalist analyzing a startup. Based on the provided information, 
        give a comprehensive analysis and predict if the startup will be successful or not.
        
        Your analysis should include:
        1. Market analysis
        2. Product/technology evaluation
        3. Founder/team assessment
        4. Overall score (1-10)
        5. Investment recommendation 
        """
        
        try:
            response = self.get_structured_output(BaselineAnalysis, prompt, startup_info_str)
            return response.dict()
            
        except Exception as e:
            self.logger.error(f"Error in baseline analysis: {str(e)}")
            return {"error": str(e)}
        
if __name__ == "__main__":
    def test_baseline_framework():
        # Create a BaselineFramework instance
        framework = BaselineFramework()
        
        # Test startup info
        test_startup = """
        Company Name: TechStart
        Description: AI-powered software development platform
        Product Details: Cloud-based IDE with AI assistance for code generation and debugging
        Technology Stack: Python, React, AWS
        Founder Background: Ex-Google engineer with 10 years experience in developer tools
        """
        
        try:
            # Test analyze_startup method
            print("Testing BaselineFramework analyze_startup:")
            print("-" * 50)
            result = framework.analyze_startup(test_startup)
            
            # Print results in a formatted way
            for key, value in result.items():
                print(f"\n{key}:")
                if isinstance(value, dict):
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"  {value}")
                    
            print("\nTest completed successfully!")
            
        except Exception as e:
            print(f"Test failed with error: {str(e)}")

    # Run the test
    test_baseline_framework()
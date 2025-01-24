import os
import sys
import logging
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from agents.base_agent import BaseAgent

class BaselineAnalysis(BaseModel):
    total_analysis: str = Field(..., description="Detailed analysis of the startup")
    score: float = Field(..., description="Overall score between 1 and 10")
    recommendation: str = Field(..., description="Recommendation: 'Successful' or 'Unsuccessful'")

# Load environment variables
load_dotenv()

class BaselineFramework(BaseAgent):
    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)

    def analyze_startup(self, startup_info: str) -> dict:
        """Simple baseline analysis using only ChatGPT"""
        self.logger.info("Starting baseline analysis")
        
        # Format startup info similar to other agents
        
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
            response = self.get_json_response(BaselineAnalysis, prompt, "Startup Info: " + startup_info)
            return response.dict()
            
        except Exception as e:
            self.logger.error(f"Error in baseline analysis: {str(e)}")
            return {"error": str(e)}

if __name__ == "__main__":
    def test_baseline_framework():
        # Create a BaselineFramework instance
        framework = BaselineFramework()
        
        # Test startup info following the pattern from FounderAgent test
        startup_info = "We are a startup that provides a platform for AI-powered software development. Our founders are from Oxford university."
        
        try:
            print("Testing BaselineFramework analyze_startup:")
            print("-" * 50)
            result = framework.analyze_startup(startup_info)
            
            print("\nAnalysis Results:")
            for key, value in result.items():
                print(f"\n{key}:")
                print(f"  {value}")
                    
            print("\nTest completed successfully!")
            
        except Exception as e:
            print(f"Test failed with error: {str(e)}")

    # Run the test
    test_baseline_framework()
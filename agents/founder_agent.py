import os
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from agents.base_agent import BaseAgent
from utils.api_wrapper import OpenAIAPI
from pydantic import BaseModel, Field

class FounderAnalysis(BaseModel):
    competency_score: int = Field(..., description="Founder competency score on a scale of 1 to 10")
    analysis: str = Field(..., description="Detailed analysis of the founding team, including strengths and challenges.")

class AdvancedFounderAnalysis(FounderAnalysis):
    segmentation: int = Field(..., description="Founder segmentation level. 1-5. 1 is L1, 5 is L5")
    cosine_similarity: float = Field(..., description="The cosine similarity between founder's desc and startup info.")
    idea_fit: float = Field(..., description="Idea fit score")

class FounderSegmentation(BaseModel):
    segmentation: int = Field(..., description="Founder segmentation level. 1-5. 1 is L1, 5 is L5")

class FounderAgent(BaseAgent):
    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model)
        self.neural_network = load_model(os.path.join(project_root, 'models', 'neural_network.keras'))

    def analyze(self, startup_info, mode):
        founder_info = self._get_founder_info(startup_info)
        
        if mode == "advanced":
            basic_analysis = self.get_json_response(FounderAnalysis, self._get_analysis_prompt(), founder_info)
            segmentation = self.segment_founder(founder_info)
            idea_fit, cosine_similarity = self.calculate_idea_fit(startup_info, founder_info)
            
            return AdvancedFounderAnalysis(
                **basic_analysis.dict(),
                segmentation=segmentation,
                cosine_similarity=cosine_similarity,
                idea_fit=idea_fit,
            )
        else:
            return self.get_json_response(FounderAnalysis, self._get_analysis_prompt(), founder_info)

    def _get_founder_info(self, startup_info):
        return f"Founders' Backgrounds: {startup_info.get('founder_backgrounds', '')}\n" \
               f"Track Records: {startup_info.get('track_records', '')}\n" \
               f"Leadership Skills: {startup_info.get('leadership_skills', '')}\n" \
               f"Vision and Alignment: {startup_info.get('vision_alignment', '')}"

    def segment_founder(self, founder_info):
        return self.get_json_response(FounderSegmentation, self._get_segmentation_prompt(), founder_info).segmentation

    def calculate_idea_fit(self, startup_info, founder_info):
        founder_embedding = self.openai_api.get_embeddings(founder_info)
        startup_embedding = self.openai_api.get_embeddings(startup_info['description'])
        cosine_sim = self._calculate_cosine_similarity(founder_embedding, startup_embedding)
        
        # Prepare input for neural network
        X_new_embeddings = np.array(founder_embedding).reshape(1, -1)
        X_new_embeddings_2 = np.array(startup_embedding).reshape(1, -1)
        X_new_cosine = np.array([[cosine_sim]])
        X_new = np.concatenate([X_new_embeddings, X_new_embeddings_2, X_new_cosine], axis=1)

        # Predict using the neural network
        idea_fit = self.neural_network.predict(X_new)[0][0]
        return float(idea_fit), cosine_sim

    def _calculate_cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1).reshape(1, -1)
        vec2 = np.array(vec2).reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]

    def _get_analysis_prompt(self):
        return """
        As a highly qualified analyst specializing in startup founder assessment, evaluate the founding team based on the provided information.
        Consider the founders' educational background, industry experience, leadership capabilities, and their ability to align and execute on the company's vision.
        Provide a competency score, key strengths, and potential challenges. Please write in great details.
        """

    def _get_segmentation_prompt(self):
        return """
        Categorize the founder into one of these levels: L1, L2, L3, L4, L5.
        L5: Entrepreneur who has built a $100M+ ARR business or had a major exit.
        L4: Entrepreneur with a small to medium-size exit or executive at a notable tech company.
        L3: 10-15 years of technical and management experience.
        L2: Entrepreneurs with a few years of experience or accelerator graduates.
        L1: Entrepreneurs with negligible experience but large potential.

        Based on the following information, determine the appropriate level:
        {founder_info}
        """

if __name__ == "__main__":
    def test_founder_agent():
        # Create a FounderAgent instance
        agent = FounderAgent()

        # Test startup info
        startup_info = {
            "founder_backgrounds": "MBA from Stanford, 5 years at Google as Product Manager",
            "track_records": "Successfully launched two products at Google, one reaching 1M users",
            "leadership_skills": "Led a team of 10 engineers and designers",
            "vision_alignment": "Strong passion for AI and its applications in healthcare",
            "description": "AI-powered health monitoring wearable device"
        }

        # Test basic analysis
        print("Basic Analysis:")
        basic_analysis = agent.analyze(startup_info, mode="basic")
        print(basic_analysis)
        print()

        # Test advanced analysis
        print("Advanced Analysis:")
        advanced_analysis = agent.analyze(startup_info, mode="advanced")
        print(advanced_analysis)

    # Run the test function
    test_founder_agent()
from agents.base_agent import BaseAgent
from algorithms.embedding import get_embeddings
from algorithms.similarity import calculate_cosine_similarity

class FounderAgent(BaseAgent):
    def __init__(self, model="gpt-4"):
        super().__init__(model)

    def analyze(self, startup_info, mode):
        founder_info = self._get_founder_info(startup_info)
        analysis = self.get_response(self._get_analysis_prompt(), founder_info)
        
        if mode == "advanced":
            segmentation = self.segment_founder(founder_info)
            idea_fit = self.calculate_idea_fit(startup_info, founder_info)
            analysis += f"\nFounder Segmentation: {segmentation}\nIdea Fit Score: {idea_fit}"
        
        return analysis

    def _get_founder_info(self, startup_info):
        return f"Founders' Backgrounds: {startup_info.get('founder_backgrounds', '')}\n" \
               f"Track Records: {startup_info.get('track_records', '')}\n" \
               f"Leadership Skills: {startup_info.get('leadership_skills', '')}\n" \
               f"Vision and Alignment: {startup_info.get('vision_alignment', '')}"

    def segment_founder(self, founder_info):
        return self.get_response(self._get_segmentation_prompt(), founder_info)

    def calculate_idea_fit(self, startup_info, founder_info):
        founder_embedding = get_embeddings(founder_info)
        startup_embedding = get_embeddings(startup_info['description'])
        return calculate_cosine_similarity(founder_embedding, startup_embedding)

    def _get_analysis_prompt(self):
        return """
        As a highly qualified analyst specializing in startup founder assessment, evaluate the founding team based on:
        {founder_info}

        Consider the founders' educational background, industry experience, leadership capabilities, and their ability to align and execute on the company's vision.
        Score the founders' competency on a scale of 1 to 10, and provide insights into their strengths and potential challenges.
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
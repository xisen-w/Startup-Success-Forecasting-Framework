import streamlit as st
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from agents.founder_agent import FounderAgent

def main():
    st.title("Founder Agent Analysis")

    # Create FounderAgent instance
    agent = FounderAgent()

    # Input fields
    st.header("Startup Information")
    founder_backgrounds = st.text_area("Founders' Backgrounds")
    track_records = st.text_area("Track Records")
    leadership_skills = st.text_area("Leadership Skills")
    vision_alignment = st.text_area("Vision and Alignment")
    startup_description = st.text_area("Startup Description")

    # Analysis mode selection
    mode = st.radio("Analysis Mode", ("Basic", "Advanced"))

    if st.button("Analyze"):
        startup_info = {
            "founder_backgrounds": founder_backgrounds,
            "track_records": track_records,
            "leadership_skills": leadership_skills,
            "vision_alignment": vision_alignment,
            "description": startup_description
        }

        if mode == "Basic":
            analysis = agent.analyze(startup_info, mode="basic")
            st.subheader("Basic Analysis Results")
            st.write(f"Competency Score: {analysis.competency_score}")
            st.write(f"Strengths: {analysis.strengths}")
            st.write(f"Challenges: {analysis.challenges}")
        else:
            analysis = agent.analyze(startup_info, mode="advanced")
            st.subheader("Advanced Analysis Results")
            st.write(f"Competency Score: {analysis.competency_score}")
            st.write(f"Strengths: {analysis.strengths}")
            st.write(f"Challenges: {analysis.challenges}")
            st.write(f"Segmentation: {analysis.segmentation}")
            st.write(f"Cosine Similarity: {analysis.cosine_similarity:.4f}")
            st.write(f"Idea Fit Score: {analysis.idea_fit:.4f}")

if __name__ == "__main__":
    main()
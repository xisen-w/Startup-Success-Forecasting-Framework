import streamlit as st
import sys
import os
import time
import traceback

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from ssff_framework import StartupFramework

def main():
    st.title("Startup Success Forecasting Framework")

    # Initialize the StartupFramework
    framework = StartupFramework()

    # Input field for startup information
    startup_info_str = st.text_area("Enter Startup Information", height=200,
                                    help="Provide a detailed description of the startup, including information about the product, market, founders, and any other relevant details.")

    if st.button("Analyze Startup"):
        if startup_info_str:
            result_placeholder = st.empty()
            result = analyze_startup_with_updates(framework, startup_info_str, result_placeholder)
            if result:
                display_final_results(result, "advanced")
            else:
                st.error("Analysis did not complete successfully. Please check the errors above.")
        else:
            st.warning("Please enter startup information before analyzing.")

def analyze_startup_with_updates(framework, startup_info_str, placeholder):
    with placeholder.container():
        st.write("### Analysis in Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_status(step, progress):
            status_text.text(f"Step: {step}")
            progress_bar.progress(progress)

        result = {}
        try:
            update_status("Parsing startup information", 0.1)
            startup_info = framework.vc_scout_agent.parse_record(startup_info_str)
            st.write("Startup info parsed")

            update_status("VCScout evaluation", 0.2)
            prediction, categorization = framework.vc_scout_agent.side_evaluate(startup_info)
            st.write(f"Initial Prediction: {prediction}")
            result['Categorical Prediction'] = prediction
            result['Categorization'] = categorization.dict()

            update_status("Market analysis", 0.3)
            market_analysis = framework.market_agent.analyze(startup_info.dict(), mode="advanced")
            st.write("Market Analysis Complete")
            result['Market Info'] = market_analysis.dict()

            update_status("Product analysis", 0.4)
            product_analysis = framework.product_agent.analyze(startup_info.dict(), mode="advanced")
            st.write("Product Analysis Complete")
            result['Product Info'] = product_analysis.dict()

            update_status("Founder analysis", 0.5)
            founder_analysis = framework.founder_agent.analyze(startup_info.dict(), mode="advanced")
            st.write("Founder Analysis Complete")
            result['Founder Info'] = founder_analysis.dict()

            update_status("Advanced founder analysis", 0.6)
            founder_segmentation = framework.founder_agent.segment_founder(startup_info.founder_backgrounds)
            founder_idea_fit = framework.founder_agent.calculate_idea_fit(startup_info.dict(), startup_info.founder_backgrounds)
            st.write("Advanced Founder Analysis Complete")
            result['Founder Segmentation'] = founder_segmentation
            result['Founder Idea Fit'] = founder_idea_fit[0]

            update_status("Integration", 0.8)
            integrated_analysis = framework.integration_agent.integrate_analyses(
                market_analysis.dict(),
                product_analysis.dict(),
                founder_analysis.dict(),
                prediction,
                mode="advanced"
            )
            st.write("Integration Complete")
            result['Final Decision'] = integrated_analysis.dict()

            update_status("Quantitative decision", 0.9)
            quant_decision = framework.integration_agent.getquantDecision(
                prediction,
                founder_idea_fit[0],
                founder_segmentation,
                integrated_analysis.dict()
            )
            st.write("Quantitative Decision Complete")
            result['Quantitative Decision'] = quant_decision.dict()

            update_status("Analysis complete", 1.0)
            st.write("Analysis Complete!")

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.write(traceback.format_exc())
        
        return result

def display_final_results(result, mode):
    st.subheader("Final Analysis Results")

    # Display Final Decision
    st.write("### Final Decision")
    final_decision = result['Final Decision']
    st.write(f"Overall Score: {final_decision['overall_score']:.2f}")
    st.write(f"Summary: {final_decision['summary']}")
    st.write("Strengths:")
    for strength in final_decision['strengths']:
        st.write(f"- {strength}")
    st.write("Weaknesses:")
    for weakness in final_decision['weaknesses']:
        st.write(f"- {weakness}")
    st.write(f"Recommendation: {final_decision['recommendation']}")

    # Display Market Info
    st.write("### Market Information")
    market_info = result['Market Info']
    st.write(f"Market Size: {market_info['market_size']}")
    st.write(f"Growth Rate: {market_info['growth_rate']}")
    st.write(f"Competition: {market_info['competition']}")
    st.write(f"Market Trends: {market_info['market_trends']}")
    st.write(f"Viability Score: {market_info['viability_score']}")

    # Display Product Info
    st.write("### Product Information")
    product_info = result['Product Info']
    st.write(f"Features Analysis: {product_info['features_analysis']}")
    st.write(f"Tech Stack Evaluation: {product_info['tech_stack_evaluation']}")
    st.write(f"USP Assessment: {product_info['usp_assessment']}")
    st.write(f"Potential Score: {product_info['potential_score']}")
    st.write(f"Innovation Score: {product_info['innovation_score']}")
    st.write(f"Market Fit Score: {product_info['market_fit_score']}")

    # Display Founder Info
    st.write("### Founder Information")
    founder_info = result['Founder Info']
    st.write(f"Competency Score: {founder_info['competency_score']}")
    st.write(f"Strengths: {founder_info['strengths']}")
    st.write(f"Challenges: {founder_info['challenges']}")

    # Display Prediction and Categorization
    st.write("### Prediction and Categorization")
    st.write(f"Prediction: {result['Categorical Prediction']}")
    st.write("Categorization:")
    for key, value in result['Categorization'].items():
        st.write(f"- {key}: {value}")

    # Display Advanced Analysis results if applicable
    if mode.lower() == "advanced":
        st.write("### Advanced Analysis")
        if 'Founder Segmentation' in result:
            st.write(f"Founder Segmentation: {result['Founder Segmentation']}")
        if 'Founder Idea Fit' in result:
            st.write(f"Founder Idea Fit: {result['Founder Idea Fit']:.4f}")
        
        if 'Quantitative Decision' in result:
            st.write("### Quantitative Decision")
            quant_decision = result['Quantitative Decision']
            st.write(f"Outcome: {quant_decision['outcome']}")
            st.write(f"Probability: {quant_decision['probability']:.4f}")
            st.write(f"Reasoning: {quant_decision['reasoning']}")

if __name__ == "__main__":
    main()
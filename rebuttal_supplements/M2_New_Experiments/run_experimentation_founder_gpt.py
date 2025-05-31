# Founder-GPT-style baseline experiment script

import os
import sys
import logging
import argparse
import pandas as pd
import json
from tqdm import tqdm
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime
import time
import numpy as np
from typing import Optional, List, Dict, Any

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from agents.base_agent import BaseAgent # Assuming BaseAgent and its OpenAIAPI wrapper are set up

# --- Pydantic Models ---

class FounderGPTLLMOutput(BaseModel):
    simulated_discussion: str = Field(..., description="The full text of the three simulated VC analysts brainstorming, critiquing, and converging on their assessment, covering founder/idea features and their likelihoods for both founder and idea aspects.")
    founder_score_eta_f: float = Field(..., description="The consensus founder score (η_f) between 0 and 1, resulting from the discussion.", ge=0, le=1)
    idea_score_eta_i: float = Field(..., description="The consensus idea score (η_i) between 0 and 1, resulting from the discussion.", ge=0, le=1)

class FounderGPTExperimentResult(BaseModel):
    original_index: int
    org_uuid: Optional[str] = None
    org_name: Optional[str] = None
    label: Optional[Any] = None # Ground truth from input CSV

    # Raw text input to LLM (for reference)
    founder_text_input: Optional[str] = None
    startup_text_input: Optional[str] = None

    # Output from LLM
    simulated_discussion: Optional[str] = None
    llm_founder_score_eta_f: Optional[float] = None
    llm_idea_score_eta_i: Optional[float] = None

    # Calculated scores
    fit_score_eta_fit: Optional[float] = None
    final_aggregated_score_eta: Optional[float] = None 

    recommendation: Optional[str] = None # "Successful" or "Unsuccessful"

    error: Optional[str] = None
    # startup_info_processed: Optional[str] = None # Replaced by specific error logging

# --- Helper Functions ---

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0.0
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if vec1.shape != vec2.shape or np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# --- Framework Class ---

class FounderGPTFramework(BaseAgent):
    def __init__(self, model="gpt-4o-mini", decision_threshold=0.25): # Max eta is 0.5
        super().__init__(model)
        self.logger = logging.getLogger(__name__)
        self.decision_threshold = decision_threshold
        # Example reference cases table (can be made more dynamic later)
        self.succ_fail_table_example = """
        ### Example Success Cases:
        1. Founder A (AI SaaS): Strong technical background, previous small exit, clear vision for a niche B2B market. Idea solved a painful, unaddressed problem.
        2. Founder B (Marketplace): Deep industry expertise, strong network, resilient through early pivots. Idea had strong network effects and clear monetization.
        3. Founder C (Deep Tech): PhD in relevant field, patented technology, strong research team. Idea was groundbreaking with high barrier to entry.

        ### Example Failure Cases:
        1. Founder X (Social App): Marketing background, no technical co-founder, chased trends. Idea was a "vitamin", not a "painkiller", in a crowded market.
        2. Founder Y (E-commerce): First-time founder, underestimated operational complexity and costs. Idea had thin margins and strong, established competitors.
        3. Founder Z (Hardware): Over-engineered product, slow to market, ran out of cash before finding product-market fit. Idea required significant capital not secured.
        """

    def analyze_startup(self, founder_text: str, startup_text: str) -> dict:
        self.logger.debug(f"Starting Founder-GPT analysis. Founder: {founder_text[:50]}..., Startup: {startup_text[:50]}...")

        system_prompt = """You are to simulate THREE seasoned venture-capital analysts collaborating.
They will brainstorm step-by-step, critique each other,
back-track on flaws, and converge on the most logical assessment.
Your entire output should be a single JSON object matching the provided schema.
"""
        user_prompt_template = '''
### INPUT
Founder profile:
{founder_profile}

Startup description:
{startup_description}

REFERENCE CASES (example): 
{reference_cases}

### TASK
Simulate the three experts performing the following steps. Capture the full discussion in the 'simulated_discussion' field of your JSON response.

STEP 1 (Brainstorm Features): 
Brainstorm 4-6 bullet "Successful Founder/Idea Features" by comparing the input to the reference cases. Show pros & cons and discuss until agreed. Document this discussion.

STEP 2 (Rate Features): 
For EACH feature identified for the FOUNDER, each expert rates "likelihood of success contribution" (0–1) and debates until consensus for that feature. Repeat for IDEA features. Document this discussion.
   Example for one founder feature:
   Expert 1 (Founder Feature X): ... likelihood: 0.Y
   Expert 2 (Founder Feature X): ... likelihood: 0.Z
   Expert 3 (Founder Feature X): ... likelihood: 0.A
   Consensus (Founder Feature X): ... likelihood: 0.B

STEP 3 (Determine Scores): 
Based on the entire discussion and feature analysis, determine and output:
   - A consensus `founder_score_eta_f` (a single number between 0 and 1).
   - A consensus `idea_score_eta_i` (a single number between 0 and 1).

The `simulated_discussion` field should contain all the detailed interactions from STEP 1, STEP 2, and the reasoning for the STEP 3 scores.
The final JSON should also include `founder_score_eta_f` and `idea_score_eta_i` as top-level fields.
'''
        
        user_prompt = user_prompt_template.format(
            founder_profile=founder_text,
            startup_description=startup_text,
            reference_cases=self.succ_fail_table_example
        )

        analysis_result = {
            "founder_text_input": founder_text,
            "startup_text_input": startup_text,
            "simulated_discussion": None,
            "llm_founder_score_eta_f": None,
            "llm_idea_score_eta_i": None,
            "fit_score_eta_fit": None,
            "final_aggregated_score_eta": None,
            "recommendation": None,
            "error": None
        }

        try:
            # 1. Get LLM analysis (discussion, eta_f, eta_i)
            llm_output = self.get_json_response(FounderGPTLLMOutput, user_prompt, system_prompt) # Swapped user & system
            if llm_output is None:
                raise ValueError("LLM returned None for FounderGPTLLMOutput.")

            analysis_result["simulated_discussion"] = llm_output.simulated_discussion
            eta_f = llm_output.founder_score_eta_f
            eta_i = llm_output.idea_score_eta_i
            analysis_result["llm_founder_score_eta_f"] = eta_f
            analysis_result["llm_idea_score_eta_i"] = eta_i

            # 2. Calculate fit_score_eta_fit
            founder_embedding = self.openai_api.get_embeddings(founder_text)
            startup_embedding = self.openai_api.get_embeddings(startup_text)
            
            if founder_embedding is None or startup_embedding is None:
                self.logger.warning("Could not generate embeddings for founder or startup text. Fit score will be 0.")
                eta_fit = 0.0
            else:
                eta_fit = cosine_similarity(founder_embedding, startup_embedding)
            analysis_result["fit_score_eta_fit"] = eta_fit

            # 3. Calculate final_aggregated_score_eta
            # η = η_f × η_fit × η_i / 2. If any component = 0 → η = 0
            if eta_f == 0.0 or eta_fit == 0.0 or eta_i == 0.0:
                final_eta = 0.0
            else:
                final_eta = (eta_f * eta_fit * eta_i) / 2.0
            analysis_result["final_aggregated_score_eta"] = final_eta
            
            # 4. Determine recommendation
            analysis_result["recommendation"] = "Successful" if final_eta >= self.decision_threshold else "Unsuccessful"

        except Exception as e:
            self.logger.error(f"Error in Founder-GPT analysis pipeline: {str(e)}", exc_info=True)
            analysis_result["error"] = str(e)
        
        return analysis_result

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Run Founder-GPT style baseline startup analysis experiments.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the input CSV dataset.")
    parser.add_argument("--founder_column", type=str, required=True, help="Name of the CSV column for founder profile text.")
    parser.add_argument("--startup_column", type=str, required=True, help="Name of the CSV column for startup description text.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", choices=["gpt-4o-mini", "gpt-4o", "o1-mini", "o3-mini"], help="LLM model to use.")
    parser.add_argument("--num_rows", type=int, default=None, help="Number of rows to process. Processes all if not specified.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output JSON file. Auto-generates if None.")
    parser.add_argument("--info_column", type=str, default="integrated_info", help="Fallback column if founder/startup specific columns are missing (not actively used if others provided).") # Kept for arg consistency for now
    parser.add_argument("--decision_threshold", type=float, default=0.25, help="Threshold for 'Successful' recommendation (max eta is 0.5).")


    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)
    # The dataset path is now expected to be a full path or relative to CWD,
    # as 'data/' subdirectory isn't enforced by default for this more flexible script
    # full_dataset_path = os.path.join(script_dir, args.dataset_path) # Old way
    full_dataset_path = args.dataset_path


    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name_part = os.path.splitext(os.path.basename(args.dataset_path))[0].replace("curated_dataset_success_", "").replace("curated_dataset_", "")
        results_dir = os.path.join(script_dir, "experiment_results")
        os.makedirs(results_dir, exist_ok=True)
        output_filename = f"founder_gpt_{args.model_name}_{dataset_name_part}_{timestamp}.json"
        args.output_file = os.path.join(results_dir, output_filename)
    else:
        output_dir_for_specified_file = os.path.dirname(args.output_file)
        if output_dir_for_specified_file:
            os.makedirs(output_dir_for_specified_file, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
    logging.info(f"Starting Founder-GPT experiment with LLM model: {args.model_name}")
    logging.info(f"Processing dataset: {full_dataset_path}")
    if args.num_rows:
        logging.info(f"Number of rows to process: {args.num_rows}")
    logging.info(f"Founder column: '{args.founder_column}', Startup column: '{args.startup_column}'")
    logging.info(f"Output will be saved to (JSON List format): {args.output_file}")
    logging.info(f"Decision threshold (eta >= threshold for Successful): {args.decision_threshold}")


    try:
        df = pd.read_csv(full_dataset_path)
        logging.info(f"Successfully loaded dataset. Total rows: {len(df)}")
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {full_dataset_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        sys.exit(1)

    required_cols = [args.founder_column, args.startup_column, 'org_uuid', 'org_name', 'label']
    for col in required_cols:
        if col not in df.columns:
            logging.error(f"Required column '{col}' not found in the dataset.")
            logging.error(f"Available columns: {df.columns.tolist()}")
            sys.exit(1)
    
    if args.num_rows:
        df_to_process = df.head(args.num_rows)
    else:
        df_to_process = df

    framework = FounderGPTFramework(model=args.model_name, decision_threshold=args.decision_threshold)
    
    all_results = []
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f_read:
                content = f_read.read()
                if content.strip():
                    all_results = json.loads(content)
                    if not isinstance(all_results, list):
                        all_results = []
                    else:
                        logging.info(f"Loaded {len(all_results)} existing results.")
        except Exception as e:
            logging.warning(f"Error reading existing results from {args.output_file}: {e}. Starting fresh.")
            all_results = []

    processed_indices = {res.get("original_index") for res in all_results if "original_index" in res}

    for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Analyzing startups (Founder-GPT)"):
        if index in processed_indices:
            logging.info(f"Skipping already processed original_index (Founder-GPT): {index}")
            continue

        time.sleep(1) # API rate limit delay

        founder_text = str(row[args.founder_column])
        startup_text = str(row[args.startup_column])
        
        base_data = {
            "original_index": index,
            "org_uuid": row.get('org_uuid'),
            "org_name": row.get('org_name'),
            "label": row.get('label')
        }
        
        current_result_data = {}
        if pd.isna(founder_text) or pd.isna(startup_text) or not founder_text.strip() or not startup_text.strip():
            logging.warning(f"Skipping row {index} due to missing founder or startup info.")
            current_result_data = {**base_data, "error": "Missing founder or startup info"}
        else:
            analysis_output = framework.analyze_startup(founder_text, startup_text)
            # Use Pydantic model to structure and validate before appending
            try:
                validated_output = FounderGPTExperimentResult(**base_data, **analysis_output)
                current_result_data = validated_output.dict()
            except Exception as pydantic_error:
                logging.error(f"Pydantic validation error for index {index}: {pydantic_error}")
                current_result_data = {**base_data, "error": f"Pydantic validation error: {pydantic_error}", **analysis_output}

        all_results.append(current_result_data)

        try:
            with open(args.output_file, 'w') as outfile:
                json.dump(all_results, outfile, indent=4, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
        except Exception as e:
            logging.error(f"CRITICAL: Could not write Founder-GPT results to {args.output_file} after index {index}. Error: {e}")

    try:
        with open(args.output_file, 'w') as outfile:
            json.dump(all_results, outfile, indent=4, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
        logging.info(f"Founder-GPT Processing complete. Results saved to {args.output_file}")
    except Exception as e:
        logging.error(f"CRITICAL: Could not write final Founder-GPT results to {args.output_file}. Error: {e}")

if __name__ == "__main__":
    main() 
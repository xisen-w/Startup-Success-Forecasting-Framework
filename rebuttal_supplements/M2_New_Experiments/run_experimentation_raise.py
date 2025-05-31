# R.A.I.S.E.-style (Stage 4 Inference) baseline experiment script

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
from typing import Optional, List, Dict, Any

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from agents.base_agent import BaseAgent

# --- Pydantic Models ---

class RAISELLMPrediction(BaseModel):
    prediction: str = Field(..., description="Prediction: HIGH or LOW likelihood of success.")
    explanation: str = Field(..., description="Brief rationale for the prediction based on the policy.")

class RAISEExperimentResult(BaseModel):
    original_index: int
    org_uuid: Optional[str] = None
    org_name: Optional[str] = None
    label: Optional[Any] = None # Ground truth from input CSV

    # Raw text input to LLM
    founder_text_input: Optional[str] = None
    startup_text_input: Optional[str] = None
    policy_text_used: Optional[str] = None

    # Output from LLM
    llm_prediction_raw: Optional[str] = None # HIGH/LOW
    llm_explanation: Optional[str] = None

    # Mapped outputs
    recommendation: Optional[str] = None # "Successful" or "Unsuccessful"
    derived_score: Optional[float] = None # e.g., HIGH=1.0, LOW=0.0

    error: Optional[str] = None

# --- Framework Class ---

class RAISEFramework(BaseAgent):
    def __init__(self, model="gpt-4o-mini", policy_text: Optional[str] = None):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)
        if policy_text:
            self.policy_text = policy_text
        else:
            self.logger.warning("No policy text provided. Using a placeholder example policy.")
            self.policy_text = """
            IF founder_has_relevant_deep_domain_expertise THEN likelihood_of_success = HIGH.
            IF founder_has_prior_successful_exit_in_same_sector THEN likelihood_of_success = HIGH.
            IF product_addresses_clear_underserved_market_need THEN likelihood_of_success = HIGH.
            IF founder_lacks_technical_skills_for_tech_product AND no_technical_cofounder THEN likelihood_of_success = LOW.
            IF market_is_highly_saturated_with_strong_incumbents AND product_is_not_differentiated THEN likelihood_of_success = LOW.
            IF founder_shows_lack_of_coachability_or_adaptability THEN likelihood_of_success = LOW.
            """
        self.logger.debug(f"RAISE Framework initialized with policy:\n{self.policy_text[:200]}...")

    def analyze_startup(self, founder_profile: str, startup_description: str) -> dict:
        self.logger.debug(f"Starting R.A.I.S.E. Stage 4 analysis. Founder: {founder_profile[:50]}..., Startup: {startup_description[:50]}...")

        system_prompt = """You are an expert startup analyst. Apply the given decision policy to the founder and startup information to predict success.
Your entire output should be a single JSON object matching the provided schema.
"""
        
        user_prompt_template = '''
Founder Profile:
{profile}

Startup Description:
{description}

Based on the decision policy below, predict whether the founder is likely to succeed.

Decision Policy:
{policy_text}

Return a JSON with "prediction" (HIGH or LOW) and "explanation" (brief rationale).
'''
        
        user_prompt = user_prompt_template.format(
            profile=founder_profile,
            description=startup_description,
            policy_text=self.policy_text
        )

        analysis_result = {
            "founder_text_input": founder_profile,
            "startup_text_input": startup_description,
            "policy_text_used": self.policy_text,
            "llm_prediction_raw": None,
            "llm_explanation": None,
            "recommendation": None,
            "derived_score": None,
            "error": None
        }

        try:
            llm_output = self.get_json_response(RAISELLMPrediction, user_prompt, system_prompt)
            if llm_output is None:
                raise ValueError("LLM returned None for RAISELLMPrediction.")

            analysis_result["llm_prediction_raw"] = llm_output.prediction
            analysis_result["llm_explanation"] = llm_output.explanation

            if llm_output.prediction.upper() == "HIGH":
                analysis_result["recommendation"] = "Successful"
                analysis_result["derived_score"] = 1.0
            elif llm_output.prediction.upper() == "LOW":
                analysis_result["recommendation"] = "Unsuccessful"
                analysis_result["derived_score"] = 0.0
            else:
                self.logger.warning(f"LLM prediction was neither HIGH nor LOW: {llm_output.prediction}. Defaulting recommendation.")
                analysis_result["recommendation"] = "Uncertain"
                analysis_result["derived_score"] = 0.5 # Or handle as error
                analysis_result["error"] = f"Unexpected prediction value: {llm_output.prediction}"

        except Exception as e:
            self.logger.error(f"Error in R.A.I.S.E. analysis pipeline: {str(e)}", exc_info=True)
            analysis_result["error"] = str(e)
        
        return analysis_result

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Run R.A.I.S.E. (Stage 4 Inference) style baseline startup analysis.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the input CSV dataset.")
    parser.add_argument("--founder_column", type=str, required=True, help="Name of the CSV column for founder profile text.")
    parser.add_argument("--startup_column", type=str, required=True, help="Name of the CSV column for startup description text.")
    parser.add_argument("--policy_file", type=str, default=None, help="Path to a text file containing the decision policy. Uses placeholder if None.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", choices=["gpt-4o-mini", "gpt-4o", "o1-mini", "o3-mini"], help="LLM model to use.")
    parser.add_argument("--num_rows", type=int, default=None, help="Number of rows to process. Processes all if not specified.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output JSON file. Auto-generates if None.")
    parser.add_argument("--info_column", type=str, default="integrated_info", help="Fallback column (not actively used if founder/startup cols provided).")

    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)
    full_dataset_path = args.dataset_path

    policy_content = None
    if args.policy_file:
        try:
            with open(args.policy_file, 'r') as pf:
                policy_content = pf.read()
            logging.info(f"Loaded decision policy from: {args.policy_file}")
        except FileNotFoundError:
            logging.error(f"Policy file not found: {args.policy_file}. Using placeholder policy.")
        except Exception as e:
            logging.error(f"Error reading policy file {args.policy_file}: {e}. Using placeholder policy.")

    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name_part = os.path.splitext(os.path.basename(args.dataset_path))[0].replace("curated_dataset_success_", "").replace("curated_dataset_", "")
        results_dir = os.path.join(script_dir, "experiment_results")
        os.makedirs(results_dir, exist_ok=True)
        output_filename = f"raise_inference_{args.model_name}_{dataset_name_part}_{timestamp}.json"
        args.output_file = os.path.join(results_dir, output_filename)
    else:
        output_dir_for_specified_file = os.path.dirname(args.output_file)
        if output_dir_for_specified_file:
            os.makedirs(output_dir_for_specified_file, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
    logging.info(f"Starting R.A.I.S.E. (Stage 4 Inference) experiment with LLM model: {args.model_name}")
    logging.info(f"Processing dataset: {full_dataset_path}")
    if args.num_rows:
        logging.info(f"Number of rows to process: {args.num_rows}")
    logging.info(f"Founder column: '{args.founder_column}', Startup column: '{args.startup_column}'")
    logging.info(f"Output will be saved to (JSON List format): {args.output_file}")

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

    framework = RAISEFramework(model=args.model_name, policy_text=policy_content)
    
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

    for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Analyzing startups (R.A.I.S.E. Inference)"):
        if index in processed_indices:
            logging.info(f"Skipping already processed original_index (R.A.I.S.E.): {index}")
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
            try:
                validated_output = RAISEExperimentResult(**base_data, **analysis_output)
                current_result_data = validated_output.dict()
            except Exception as pydantic_error:
                logging.error(f"Pydantic validation error for R.A.I.S.E. index {index}: {pydantic_error}")
                current_result_data = {**base_data, "error": f"Pydantic validation error: {pydantic_error}", **analysis_output}
        
        all_results.append(current_result_data)

        try:
            with open(args.output_file, 'w') as outfile:
                json.dump(all_results, outfile, indent=4, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
        except Exception as e:
            logging.error(f"CRITICAL: Could not write R.A.I.S.E. results to {args.output_file} after index {index}. Error: {e}")

    try:
        with open(args.output_file, 'w') as outfile:
            json.dump(all_results, outfile, indent=4, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
        logging.info(f"R.A.I.S.E. (Stage 4 Inference) Processing complete. Results saved to {args.output_file}")
    except Exception as e:
        logging.error(f"CRITICAL: Could not write final R.A.I.S.E. results to {args.output_file}. Error: {e}")

if __name__ == "__main__":
    main() 
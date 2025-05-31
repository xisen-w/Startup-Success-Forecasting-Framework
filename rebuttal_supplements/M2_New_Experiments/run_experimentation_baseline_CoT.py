# This file is bult on the previous fle for a detailed experimentation.ExceptionGroup
    
#Essentially, we do this for each experiment. 

# 4o-mini, 4o, o1-mini, o3-mini. (four groups of models)

# and then for each model, we are basically calling ssff framework but here's the thing

# we also have baseline methods (intotal we compare 6 methods): method 1 pure vannila GPT. method 2: COT prompting. 3: FoundersGPT 4: RAISE Paper. 5.Without ML 6.With ML.

# But for this draft. we focus on the pure vanilla GPT

# We also like the output to be like jsonl and also use tqdm to show the progress. We want after each round, we save another line. 

# The data sets are like: rebuttal_supplements/M2_New_Experiments/data/curated_dataset_success_10pct.csv to 50pct now. try to make it that we can use the command to choose which dataset to use. the 10pct is the default.

# also enable in the command line to choose the model (while multiple models are supported) and then also support :n (like how many lines of input do we processs)

import os
import sys
import logging
import argparse
import pandas as pd
import json
from tqdm import tqdm
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime # Import datetime
import time # Import time for adding delays

# Add the project root directory to the Python path
# Assuming this script is in rebuttal_supplements/M2_New_Experiments/
# and the 'agents' directory is at the project root level,
# which is two levels up from this script's current directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# ---- START DEBUGGING BLOCK ----
print(f"DEBUG: Current sys.path: {sys.path}")
agents_path = os.path.join(project_root, 'agents')
print(f"DEBUG: Expected 'agents' directory path: {agents_path}")
print(f"DEBUG: Does 'agents' directory exist? {os.path.isdir(agents_path)}")
if os.path.isdir(agents_path):
    print(f"DEBUG: Contents of 'agents' directory: {os.listdir(agents_path)}")
    init_py_path = os.path.join(agents_path, '__init__.py')
    print(f"DEBUG: Expected '__init__.py' path: {init_py_path}")
    print(f"DEBUG: Does '__init__.py' exist in 'agents'? {os.path.isfile(init_py_path)}")
# ---- END DEBUGGING BLOCK ----

from agents.base_agent import BaseAgent

class BaselineAnalysis(BaseModel):
    total_analysis: str = Field(..., description="Detailed analysis of the startup, summarizing the step-by-step reasoning")
    score: float = Field(..., description="Overall score between 1 and 10, derived from the step-by-step analysis")
    recommendation: str = Field(..., description="Recommendation: 'Successful' or 'Unsuccessful', based on the overall analysis and score")

# Load environment variables
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaselineCoTFramework(BaseAgent): # Renamed class for clarity
    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)

    def analyze_startup(self, startup_info: str) -> dict:
        """Baseline analysis using ChatGPT with Chain-of-Thought prompting"""
        self.logger.debug(f"Starting CoT baseline analysis for: {startup_info[:100]}...") 
        
        prompt = """
        You are an experienced venture capitalist analyzing a startup. Based on the provided information, 
        please think step-by-step to give a comprehensive analysis and predict if the startup will be successful or not.

        Your step-by-step thinking process should lead to the following final outputs:
        1. Market analysis (Consider TAM, SAM, SOM, competition, market trends, and barriers to entry)
        2. Product/technology evaluation (Consider innovation, scalability, defensibility, and product-market fit)
        3. Founder/team assessment (Consider experience, execution ability, team completeness, and advisor quality)
        4. Overall score (1-10, derived from your step-by-step analysis)
        5. Investment recommendation (must be 'Successful' or 'Unsuccessful', based on your overall analysis and score)

         Criteria for future success (like the destination of the startup in the future according to your prediction): 
        - Startups that raised more than $500M, acquired more than $500M or had an initial public offering over $500M valuation are defined as success. Startups that raised between $100K and $4M but did not achieve significant success afterwards are considered as failed.
        
        Please ensure your final JSON output strictly adheres to the schema with 'total_analysis', 'score', and 'recommendation'. The 'total_analysis' field should summarize your step-by-step reasoning.
        """
        
        try:
            response = self.get_json_response(BaselineAnalysis, prompt, startup_info) # Schema remains BaselineAnalysis
            return response.dict()
            
        except Exception as e:
            self.logger.error(f"Error in CoT baseline analysis for startup info '{startup_info[:100]}...': {str(e)}")
            return {"error": str(e), "startup_info_processed": startup_info}

def main():
    parser = argparse.ArgumentParser(description="Run Chain-of-Thought baseline startup analysis experiments.")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="data/curated_dataset_success_10pct.csv",
        help="Path to the input CSV dataset, relative to the script's directory's 'data' subdirectory."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="gpt-4o-mini",
        choices=["gpt-4o-mini", "gpt-4o", "o1-mini", "o3-mini"], 
        help="LLM model to use for analysis."
    )
    parser.add_argument(
        "--num_rows", 
        type=int, 
        default=None,
        help="Number of rows to process from the dataset. Processes all if not specified."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None,
        help="Path to the output JSON file. If not specified, a name with timestamp will be generated in 'experiment_results/'."
    )
    parser.add_argument(
        "--info_column",
        type=str,
        default="description", 
        help="Name of the column in the CSV that contains the startup information text."
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(__file__) 
    full_dataset_path = os.path.join(script_dir, args.dataset_path)

    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name_part = os.path.splitext(os.path.basename(args.dataset_path))[0].replace("curated_dataset_success_", "")
        results_dir = os.path.join(script_dir, "experiment_results")
        os.makedirs(results_dir, exist_ok=True)
        # Add _CoT to differentiate output files
        output_filename = f"baseline_CoT_{args.model_name}_{dataset_name_part}_{timestamp}.json" 
        args.output_file = os.path.join(results_dir, output_filename)
    else:
        output_dir_for_specified_file = os.path.dirname(args.output_file)
        if output_dir_for_specified_file: 
            os.makedirs(output_dir_for_specified_file, exist_ok=True)

    logging.info(f"Starting CoT experiment with an LLM model: {args.model_name}")
    logging.info(f"Processing dataset: {full_dataset_path}")
    if args.num_rows:
        logging.info(f"Number of rows to process: {args.num_rows}")
    logging.info(f"Output will be saved to (JSON List format): {args.output_file}")
    logging.info(f"Startup info will be read from column: '{args.info_column}'")

    try:
        df = pd.read_csv(full_dataset_path)
        logging.info(f"Successfully loaded dataset. Total rows: {len(df)}")
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {full_dataset_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        sys.exit(1)

    if args.info_column not in df.columns:
        logging.error(f"Specified info_column '{args.info_column}' not found in the dataset.")
        logging.error(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    if args.num_rows:
        df_to_process = df.head(args.num_rows)
    else:
        df_to_process = df

    framework = BaselineCoTFramework(model=args.model_name) # Use the new class name
    
    carry_over_columns = ['org_uuid', 'org_name', 'label'] 

    for col_name in carry_over_columns:
        if col_name not in df_to_process.columns:
            logging.error(f"Specified carry-over column '{col_name}' not found in the dataset. This column is required.")
            logging.error(f"Available columns: {df_to_process.columns.tolist()}")
            sys.exit(1)

    all_results = [] 

    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f_read:
                content = f_read.read()
                if content.strip(): 
                    all_results = json.loads(content)
                    if not isinstance(all_results, list):
                        logging.warning(f"Existing output file {args.output_file} does not contain a JSON list. Starting fresh.")
                        all_results = []
                    else:
                        logging.info(f"Loaded {len(all_results)} existing results from {args.output_file}")
        except json.JSONDecodeError:
            logging.warning(f"Could not decode JSON from existing file {args.output_file}. Starting fresh.")
            all_results = []
        except Exception as e:
            logging.error(f"Error reading existing results file {args.output_file}: {e}. Starting fresh.")
            all_results = []

    for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Analyzing startups (CoT)"):
        if any(res.get("original_index") == index for res in all_results):
            logging.info(f"Skipping already processed original_index (CoT): {index}")
            continue

        time.sleep(1) 

        startup_info_text = row[args.info_column]
        base_data = {"original_index": index}
        for col_name in carry_over_columns:
            if col_name in row:
                base_data[col_name] = row[col_name]
            else:
                base_data[col_name] = None

        current_result_data = {}
        if pd.isna(startup_info_text):
            logging.warning(f"Skipping row {index} due to missing startup info in column '{args.info_column}' (CoT).")
            current_result_data = {**base_data, "error": "Missing startup info"}
        else:
            llm_analysis_result = framework.analyze_startup(str(startup_info_text))
            current_result_data = {**base_data, **llm_analysis_result}
        
        all_results.append(current_result_data)

        try:
            with open(args.output_file, 'w') as outfile:
                json.dump(all_results, outfile, indent=4) 
        except Exception as e:
            logging.error(f"CRITICAL: Could not write CoT results to {args.output_file} after processing index {index}. Error: {e}")

    try:
        with open(args.output_file, 'w') as outfile:
            json.dump(all_results, outfile, indent=4)
        logging.info(f"CoT Processing complete. Results saved to {args.output_file}")
    except Exception as e:
        logging.error(f"CRITICAL: Could not write final CoT results to {args.output_file}. Error: {e}")
        logging.info(f"Data for this CoT run (may be incomplete if script interrupted before last write): {len(all_results)} records processed.")

if __name__ == "__main__":
    main()
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

# Ensure the agents module can be found
# If agents.base_agent is still not found, you might need to adjust the path
# or ensure __init__.py files are present in relevant directories.
# try:
from agents.base_agent import BaseAgent
# except ImportError as e: # Capture the exception instance
#     print(f"Error: Could not import BaseAgent. Import error was: {e}") # Print the specific import error
#     print(f"Current sys.path for import: {sys.path}") # Re-print sys.path at point of failure
#     print(f"Project root used for sys.path.insert: {project_root}")
#     sys.exit(1)


class BaselineAnalysis(BaseModel):
    total_analysis: str = Field(..., description="Detailed analysis of the startup")
    score: float = Field(..., description="Overall score between 1 and 10")
    recommendation: str = Field(..., description="Recommendation: 'Successful' or 'Unsuccessful'")

# Load environment variables
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaselineFramework(BaseAgent):
    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)

    def analyze_startup(self, startup_info: str) -> dict:
        """Simple baseline analysis using only ChatGPT"""
        self.logger.debug(f"Starting baseline analysis for: {startup_info[:100]}...") # Log snippet
        
        prompt = """
        You are an experienced venture capitalist analyzing a startup. Based on the provided information, 
        give a comprehensive analysis and give your investment recommendation. Looking at the potential, will you invest? 
        
        Your analysis should include:
        1. Market analysis
        2. Product/technology evaluation
        3. Founder/team assessment
        4. Overall score (1-10)
        5. Investment recommendation (must be 'Successful' or 'Unsuccessful')

        Criteria for future success (like the destination of the startup in the future according to your prediction): 
        - Startups that raised more than $500M, acquired more than $500M or had an initial public offering over $500M valuation are defined as success. Startups that raised between $100K and $4M but did not achieve significant success afterwards are considered as failed.
        """
        
        try:
            # Assuming startup_info is the raw text description
            response = self.get_json_response(BaselineAnalysis, prompt, startup_info)
            return response.dict()
            
        except Exception as e:
            self.logger.error(f"Error in baseline analysis for startup info '{startup_info[:100]}...': {str(e)}")
            return {"error": str(e), "startup_info_processed": startup_info}

def main():
    parser = argparse.ArgumentParser(description="Run baseline startup analysis experiments.")
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
        choices=["gpt-4o-mini", "gpt-4o", "o1-mini", "o3-mini"], # As per your comments
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
        default="description", # Assuming 'description' is the column with startup text
        help="Name of the column in the CSV that contains the startup information text."
    )

    args = parser.parse_args()

    # Construct full dataset path
    script_dir = os.path.dirname(__file__) # Directory of the current script
    full_dataset_path = os.path.join(script_dir, args.dataset_path)

    if args.output_file is None:
        # Generate a filename with timestamp and model name, inside 'experiment_results'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name_part = os.path.splitext(os.path.basename(args.dataset_path))[0].replace("curated_dataset_success_", "") # e.g., 10pct
        
        # Define the target directory for results
        results_dir = os.path.join(script_dir, "experiment_results")
        
        # Ensure the results_dir exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Construct the full output file path
        output_filename = f"baseline_{args.model_name}_{dataset_name_part}_{timestamp}.json"
        args.output_file = os.path.join(results_dir, output_filename)
    else:
        # If an output_file is specified, ensure its directory exists
        # (This part was already good, but good to keep in mind if output_file could be an absolute path)
        output_dir_for_specified_file = os.path.dirname(args.output_file)
        if output_dir_for_specified_file: # Only if there's a directory part
            os.makedirs(output_dir_for_specified_file, exist_ok=True)


    logging.info(f"Starting experiment with an LLM model: {args.model_name}")
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

    framework = BaselineFramework(model=args.model_name)
    
    # The os.makedirs call for the output file's directory is now handled
    # when args.output_file is constructed or if it's explicitly provided.
    # So the specific block here can be simplified or rely on the logic above.
    # output_dir = os.path.dirname(args.output_file)
    # if output_dir: 
    #     os.makedirs(output_dir, exist_ok=True)
    # This is now handled correctly whether output_file is generated or user-specified.

    # Define the columns you want to carry over to the output
    carry_over_columns = ['org_uuid', 'org_name', 'label'] 

    # Verify that these carry_over_columns exist in the DataFrame
    for col_name in carry_over_columns:
        if col_name not in df_to_process.columns:
            logging.error(f"Specified carry-over column '{col_name}' not found in the dataset. This column is required.")
            logging.error(f"Available columns: {df_to_process.columns.tolist()}")
            sys.exit(1)

    all_results = [] # Initialize an empty list to store all results

    # Try to load existing results if the file exists and is valid JSON
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f_read:
                content = f_read.read()
                if content.strip(): # Check if file is not empty
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

    for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Analyzing startups"):
        # Skip already processed records if resuming (simple check based on original_index)
        # This is a basic resume functionality. More robust resume would need to check org_uuid.
        if any(res.get("original_index") == index for res in all_results):
            logging.info(f"Skipping already processed original_index: {index}")
            continue

        # Introduce a delay to avoid hitting rate limits
        time.sleep(1) # Wait for 1 second before making the API call

        startup_info_text = row[args.info_column]
        base_data = {"original_index": index}
        for col_name in carry_over_columns:
            if col_name in row:
                base_data[col_name] = row[col_name]
            else:
                base_data[col_name] = None

        current_result_data = {}
        if pd.isna(startup_info_text):
            logging.warning(f"Skipping row {index} due to missing startup info in column '{args.info_column}'.")
            current_result_data = {**base_data, "error": "Missing startup info"}
        else:
            llm_analysis_result = framework.analyze_startup(str(startup_info_text))
            current_result_data = {**base_data, **llm_analysis_result}
        
        all_results.append(current_result_data) # Append current result to the list

        # Write the entire updated list back to the file after each result
        try:
            with open(args.output_file, 'w') as outfile:
                json.dump(all_results, outfile, indent=4) # Use indent for readability
        except Exception as e:
            logging.error(f"CRITICAL: Could not write results to {args.output_file} after processing index {index}. Error: {e}")
            # Decide how to handle this: maybe save current_result_data to a temp file, or try again?
            # For now, it will just continue, and the next successful write will save all_results up to that point.

    # Final write (redundant if last iteration's write was successful, but safe)
    try:
        with open(args.output_file, 'w') as outfile:
            json.dump(all_results, outfile, indent=4)
        logging.info(f"Processing complete. Results saved to {args.output_file}")
    except Exception as e:
        logging.error(f"CRITICAL: Could not write final results to {args.output_file}. Error: {e}")
        logging.info(f"Data for this run (may be incomplete if script interrupted before last write): {len(all_results)} records processed.")

if __name__ == "__main__":
    main()
import sys
import os
import json
import pandas as pd
from tqdm import tqdm
import logging
import traceback
# import warnings # Not strictly needed if we handle sklearn well or it's not used directly here
from datetime import datetime
# from sklearn.exceptions import InconsistentVersionWarning # No sklearn direct use anticipated in this refactor
# import csv # Will be replaced by JSON list output -> This was already commented, good.
import argparse
import time
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# Suppress sklearn version warnings - Keep if StartupFramework indirectly uses it and causes warnings
# warnings.filterwarnings("ignore", category=InconsistentVersionWarning) # Commenting this out

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from ssff_framework import StartupFramework

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models for SSFF Regular/Advanced Output ---

class QuantitativeDecisionRegular(BaseModel): # Renamed to avoid conflict if imported elsewhere, though locally scoped
    decision: Optional[str] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None

class SSFFRegularExperimentResult(BaseModel):
    original_index: int
    org_uuid: Optional[str] = None
    org_name: Optional[str] = None
    label: Optional[Any] = None

    # Input Data
    founder_text_input: Optional[str] = None
    startup_text_input: Optional[str] = None
    combined_input_length: Optional[int] = None 

    # Structured Analysis components from analyze_startup()
    market_analysis_structured: Optional[Dict[str, Any]] = None
    product_analysis_structured: Optional[Dict[str, Any]] = None
    founder_analysis_structured: Optional[Dict[str, Any]] = None 
    final_analysis_pro_structured: Optional[Dict[str, Any]] = None
    basic_analysis_structured: Optional[Dict[str, Any]] = None # NEW

    # Key metrics & other info from analyze_startup()
    founder_segmentation_value: Optional[Any] = None
    founder_idea_fit_score: Optional[float] = None
    categorical_prediction_vc_scout: Optional[str] = None
    categorization_details_vc_scout: Optional[Dict[str, Any]] = None
    quantitative_decision_details: Optional[QuantitativeDecisionRegular] = None
    parsed_startup_info: Optional[Dict[str, Any]] = None
    
    error: Optional[str] = None

def extract_dict_data(prefix, data_dict):
    """Helper function to dynamically extract all keys from a dictionary with a prefix"""
    if not isinstance(data_dict, dict):
        return {f"{prefix}": data_dict}
    
    # Ensure keys in the returned dict are strings, especially for nested structures
    flat_dict = {}
    for key, value in data_dict.items():
        # If value is a dictionary itself, you might want to flatten it further or handle it
        # For now, just prefixing the top-level keys from data_dict
        flat_dict[f"{prefix}_{str(key)}"] = value 
    return flat_dict

def main():
    parser = argparse.ArgumentParser(description="Run SSFF Regular (Advanced) baseline startup analysis.") # Updated description
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the input CSV dataset.")
    parser.add_argument("--founder_column", type=str, default="paragraph", help="Name of the CSV column for founder profile text.")
    parser.add_argument("--startup_column", type=str, default="long_description", help="Name of the CSV column for startup description text.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", choices=["gpt-4o-mini", "gpt-4o", "o1-mini", "o3-mini", "gpt-4"], help="LLM model to use.")
    parser.add_argument("--num_rows", type=int, default=None, help="Number of rows to process. Processes all if not specified.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output JSON file. Auto-generates if None.")
    
    args = parser.parse_args()

    # Setup logging (can be moved to top if preferred, but here it's after arg parsing)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
    
    script_dir = os.path.dirname(__file__)
    full_dataset_path = args.dataset_path # Assume full path or relative to CWD

    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name_part = os.path.splitext(os.path.basename(args.dataset_path))[0].replace("curated_dataset_success_", "").replace("curated_dataset_", "")
        results_dir = os.path.join(script_dir, "experiment_results") # Standardized output dir
        os.makedirs(results_dir, exist_ok=True)
        output_filename = f"ssff_regular_{args.model_name}_{dataset_name_part}_{timestamp}.json" # Changed filename prefix
        args.output_file = os.path.join(results_dir, output_filename)
    else:
        # Ensure directory for specified output file exists
        output_dir_for_specified_file = os.path.dirname(args.output_file)
        if output_dir_for_specified_file: # If it's not an empty string (i.e., not just a filename in CWD)
            os.makedirs(output_dir_for_specified_file, exist_ok=True)

    logger.info(f"Starting SSFF Regular (Advanced) experiment with LLM model: {args.model_name}") # Updated log
    logger.info(f"Processing dataset: {full_dataset_path}")
    if args.num_rows:
        logger.info(f"Number of rows to process: {args.num_rows}")
    logger.info(f"Founder column: '{args.founder_column}', Startup column: '{args.startup_column}'")
    logger.info(f"Output will be saved to (JSON List format): {args.output_file}")

    try:
        df = pd.read_csv(full_dataset_path)
        logger.info(f"Successfully loaded dataset. Total rows: {len(df)}")
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {full_dataset_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)

    # Standardized columns to look for, similar to other scripts
    # For this script, we primarily need founder and startup text.
    # org_uuid, org_name, label are for consistent output.
    # The original script used 'company_name' and 'success'. We'll try to map.
    
    required_input_cols = [args.founder_column, args.startup_column]
    output_id_cols = ['org_uuid', 'org_name', 'label'] # For consistent output structure
    
    for col in required_input_cols:
        if col not in df.columns:
            logger.error(f"Required input column '{col}' not found in the dataset.")
            logger.error(f"Available columns: {df.columns.tolist()}")
            sys.exit(1)

    if args.num_rows:
        df_to_process = df.head(args.num_rows)
    else:
        df_to_process = df

    framework = StartupFramework(args.model_name)
    
    all_results = []
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f_read:
                content = f_read.read()
                if content.strip(): # Ensure file is not empty
                    all_results = json.loads(content)
                    if not isinstance(all_results, list): # Ensure it's a list
                        logger.warning(f"Output file {args.output_file} did not contain a JSON list. Starting fresh.")
                        all_results = []
                    else:
                        logger.info(f"Loaded {len(all_results)} existing results from {args.output_file}.")
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON from {args.output_file}. Starting fresh.")
            all_results = []
        except Exception as e:
            logger.warning(f"Error reading existing results from {args.output_file}: {e}. Starting fresh.")
            all_results = []
            
    processed_indices = {res.get("original_index") for res in all_results if isinstance(res, dict) and "original_index" in res}

    for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Processing startups (SSFF-Regular)"): # Updated desc
        original_df_index = row.name # Get the original DataFrame index
        if original_df_index in processed_indices:
            logger.info(f"Skipping already processed original_index (SSFF-Regular): {original_df_index}") # Updated log
            continue

        time.sleep(1) # API rate limit delay

        founder_text = str(row[args.founder_column]) if pd.notna(row[args.founder_column]) else ""
        startup_text = str(row[args.startup_column]) if pd.notna(row[args.startup_column]) else ""
        
        # Map company name and success label, preferring standardized names if available
        current_org_name = row.get('org_name', row.get('company_name', f"Unnamed Company {original_df_index}"))
        current_label = row.get('label', row.get('success'))
        current_org_uuid = row.get('org_uuid')

        base_data = {
            "original_index": original_df_index,
            "org_uuid": current_org_uuid,
            "org_name": current_org_name,
            "label": current_label,
            "founder_text_input": founder_text,
            "startup_text_input": startup_text,
        }
        
        current_result_dict = {**base_data, "error": None} # Start with base data

        if not founder_text.strip() and not startup_text.strip(): # If both are empty, less useful to analyze
            logger.warning(f"Skipping row {original_df_index} due to missing founder and startup info.")
            current_result_dict["error"] = "Missing founder and startup info"
        else:
            startup_info_str = f"""
            Company: {current_org_name} 
            Startup Description: {startup_text}
            Founder Background: {founder_text}
            """
            current_result_dict["combined_input_length"] = len(startup_info_str)

            try:
                # Run REGULAR/ADVANCED analysis
                analysis_raw_output = framework.analyze_startup(startup_info_str) # Changed to analyze_startup()
                
                quant_decision_raw = analysis_raw_output.get('Quantitative Decision', {})

                current_result_dict.update({
                    "market_analysis_structured": analysis_raw_output.get('Market Analysis'),
                    "product_analysis_structured": analysis_raw_output.get('Product Analysis'),
                    "founder_analysis_structured": analysis_raw_output.get('Founder Analysis'),
                    "final_analysis_pro_structured": analysis_raw_output.get('Final Analysis'),
                    "basic_analysis_structured": analysis_raw_output.get('Basic Analysis'), # NEW
                    
                    "founder_segmentation_value": analysis_raw_output.get('Founder Segmentation'),
                    "founder_idea_fit_score": analysis_raw_output.get('Founder Idea Fit'),
                    "categorical_prediction_vc_scout": analysis_raw_output.get('Categorical Prediction'),
                    "categorization_details_vc_scout": analysis_raw_output.get('Categorization'),
                    "parsed_startup_info": analysis_raw_output.get('Startup Info'),
                    
                    "quantitative_decision_details": QuantitativeDecisionRegular(
                        decision=quant_decision_raw.get('decision'),
                        confidence=quant_decision_raw.get('confidence'),
                        rationale=quant_decision_raw.get('rationale')
                    ).model_dump() if isinstance(quant_decision_raw, dict) else None
                })

            except Exception as e:
                logger.error(f"Error processing startup {original_df_index} ({current_org_name}): {str(e)}")
                logger.error(traceback.format_exc())
                current_result_dict["error"] = str(e)
        
        # Validate with Pydantic model before adding to results
        try:
            validated_result = SSFFRegularExperimentResult(**current_result_dict) # Use new Pydantic model
            all_results.append(validated_result.model_dump(exclude_none=True))
        except Exception as pydantic_error:
            logger.error(f"Pydantic validation error for index {original_df_index}: {pydantic_error}. Storing with error.")
            # Ensure base_data and error are present even if validation fails
            error_augmented_data = {**base_data, **current_result_dict} # merge, current_result_dict might have analysis fields
            
            existing_error = current_result_dict.get("error")
            pydantic_error_str = f"Pydantic validation error: {pydantic_error}"
            if existing_error is None:
                error_augmented_data["error"] = pydantic_error_str
            else:
                error_augmented_data["error"] = str(existing_error) + f" | {pydantic_error_str}"
            all_results.append(error_augmented_data)


        # Incrementally save to JSON
        try:
            with open(args.output_file, 'w') as outfile:
                json.dump(all_results, outfile, indent=4)
        except Exception as e:
            logger.error(f"CRITICAL: Could not write SSFF-Regular results to {args.output_file} after index {original_df_index}. Error: {e}")
            # Consider a backup save mechanism or exiting if writes fail repeatedly
    
    # Final save (optional if incremental is robust, but good for completeness)
    try:
        with open(args.output_file, 'w') as outfile:
            json.dump(all_results, outfile, indent=4)
        logger.info(f"SSFF Regular (Advanced) processing complete. Results saved to {args.output_file}") # Updated log
    except Exception as e:
        logger.error(f"CRITICAL: Could not write final SSFF-Regular results to {args.output_file}. Error: {e}")

if __name__ == "__main__":
    main()
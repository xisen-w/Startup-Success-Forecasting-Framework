import json
import sys

def find_misclassified_examples(file_path, model_type):
    """
    Analyzes a JSON results file to find one False Positive (FP) and one False Negative (FN) example.

    Args:
        file_path (str): Path to the JSON file.
        model_type (str): Type of the model ('vanilla', 'foundergpt', 'ssff_pro').
                          This determines how predictions are extracted.
    """
    found_fp = False
    found_fn = False
    fp_example = None
    fn_example = None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Attempt to load as a single JSON array
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    print(f"Error: Expected a list of JSON objects in {file_path}, but got {type(data)}. Trying line-by-line.", file=sys.stderr)
                    data = [] # Reset data if not a list
                    f.seek(0) # Go back to the beginning of the file
                    # Fall through to line-by-line processing if not a valid single JSON array
            except json.JSONDecodeError as e:
                print(f"Initial JSON load failed: {e}. Attempting to read line-by-line for newline-delimited JSON.", file=sys.stderr)
                f.seek(0) # Go back to the beginning of the file
                data = []
            
            if not data: # If initial load failed or was not a list
                print(f"Processing {file_path} as newline-delimited JSON objects.", file=sys.stderr)
                f.seek(0)
                temp_data = []
                for line_number, line in enumerate(f):
                    line = line.strip()
                    if line.startswith('[') and line_number == 0: # Skip initial array bracket if present
                        line = line[1:]
                    if line.endswith(']') and not line.startswith('['): # Skip final array bracket if present
                        line = line[:-1]
                    if line.endswith(','): # Remove trailing comma if it's an item in a list
                        line = line[:-1]
                    if line: # Ensure line is not empty
                        try:
                            temp_data.append(json.loads(line))
                        except json.JSONDecodeError as e_line:
                            print(f"Warning: Could not decode JSON object on line {line_number + 1} in {file_path}. Error: {e_line}. Line content: '{line[:100]}...'", file=sys.stderr)
                            continue
                data = temp_data
            
            if not data:
                print(f"Error: No data could be loaded from {file_path}. Please check file format.", file=sys.stderr)
                return


        for record_index, record in enumerate(data):
            if found_fp and found_fn:
                break # Found both, no need to process further

            if not isinstance(record, dict):
                print(f"Warning: Skipping record at index {record_index} as it is not a dictionary. Record: {str(record)[:100]}...", file=sys.stderr)
                continue

            true_label = record.get("label")
            org_name = record.get("org_name", "N/A")

            if true_label is None:
                # print(f"Debug: Skipping record for {org_name} due to missing 'label'.", file=sys.stderr)
                continue 

            predicted_outcome_raw = None
            analysis_text = "N/A"

            if model_type == 'vanilla':
                predicted_outcome_raw = record.get("recommendation")
                analysis_text = record.get("total_analysis", "N/A")
            elif model_type == 'foundergpt':
                predicted_outcome_raw = record.get("recommendation")
                analysis_text = record.get("simulated_discussion", "N/A")
            elif model_type == 'cot':
                predicted_outcome_raw = record.get("recommendation")
                analysis_text = record.get("total_analysis", "N/A")
            elif model_type == 'ssff_basic':
                basic_analysis = record.get("basic_analysis_structured")
                if basic_analysis and isinstance(basic_analysis, dict):
                    predicted_outcome_raw = basic_analysis.get("outcome")
                    analysis_text = basic_analysis.get("IntegratedAnalysis", "N/A")
                # else:
                    # print(f"Debug: Missing 'basic_analysis_structured' or not a dict for {org_name}.", file=sys.stderr)
            elif model_type == 'ssff_pro':
                pro_analysis = record.get("final_analysis_pro_structured")
                if pro_analysis and isinstance(pro_analysis, dict):
                    predicted_outcome_raw = pro_analysis.get("outcome")
                    analysis_text = pro_analysis.get("IntegratedAnalysis", "N/A")
                # else:
                    # print(f"Debug: Missing 'final_analysis_pro_structured' or not a dict for {org_name}.", file=sys.stderr)
            elif model_type == 'ssff_nl_basic':
                nl_basic_analysis = record.get("basic_analysis_details")
                if nl_basic_analysis and isinstance(nl_basic_analysis, dict):
                    predicted_outcome_raw = nl_basic_analysis.get("outcome")
                    analysis_text = nl_basic_analysis.get("IntegratedAnalysis", "N/A")
            elif model_type == 'ssff_nl_pro':
                nl_pro_analysis = record.get("final_analysis_pro_details")
                if nl_pro_analysis and isinstance(nl_pro_analysis, dict):
                    predicted_outcome_raw = nl_pro_analysis.get("outcome")
                    analysis_text = nl_pro_analysis.get("IntegratedAnalysis", "N/A")
            elif model_type == 'raise':
                predicted_outcome_raw = record.get("recommendation") 
                analysis_text = record.get("reasoning", record.get("analysis", "N/A"))
            else:
                print(f"Unknown model type: {model_type}", file=sys.stderr)
                return

            if predicted_outcome_raw is None:
                # print(f"Debug: Skipping record for {org_name} due to missing prediction.", file=sys.stderr)
                continue 

            predicted_label = None
            if model_type in ['vanilla', 'foundergpt', 'cot', 'raise']:
                if predicted_outcome_raw == "Successful":
                    predicted_label = 1
                elif predicted_outcome_raw == "Unsuccessful":
                    predicted_label = 0
            elif model_type in ['ssff_pro', 'ssff_basic', 'ssff_nl_basic', 'ssff_nl_pro']:
                if predicted_outcome_raw == "Invest":
                    predicted_label = 1
                elif predicted_outcome_raw in ["Hold", "Pass", "Unsuccessful"]: 
                    predicted_label = 0
            
            # print(f"Debug: {org_name}, True: {true_label}, PredRaw: {predicted_outcome_raw}, PredNorm: {predicted_label}", file=sys.stderr)


            if predicted_label is None:
                # print(f"Debug: Skipping record for {org_name} as prediction couldn't be normalized: {predicted_outcome_raw}", file=sys.stderr)
                continue

            # Check for False Positive
            if not found_fp and true_label == 0 and predicted_label == 1:
                fp_example = {
                    "org_name": org_name,
                    "true_label": true_label,
                    "predicted_raw": predicted_outcome_raw,
                    "analysis": analysis_text[:500] + "..." if analysis_text and len(analysis_text) > 500 else analysis_text
                }
                found_fp = True
                # print(f"Debug: Found FP for {org_name}", file=sys.stderr)


            # Check for False Negative
            if not found_fn and true_label == 1 and predicted_label == 0:
                fn_example = {
                    "org_name": org_name,
                    "true_label": true_label,
                    "predicted_raw": predicted_outcome_raw,
                    "analysis": analysis_text[:500] + "..." if analysis_text and len(analysis_text) > 500 else analysis_text
                }
                found_fn = True
                # print(f"Debug: Found FN for {org_name}", file=sys.stderr)

        print(f"--- Results for {model_type.upper()} from {file_path} ---")
        if fp_example:
            print("\\nFalse Positive (FP) Example:")
            print(f"  Organization: {fp_example['org_name']}")
            print(f"  True Label: {fp_example['true_label']} (Unsuccessful)")
            print(f"  Predicted: {fp_example['predicted_raw']} (Interpreted as Successful)")
            print(f"  Analysis Snippet: {fp_example['analysis']}")
        else:
            print("\\nNo False Positive example found.")

        if fn_example:
            print("\\nFalse Negative (FN) Example:")
            print(f"  Organization: {fn_example['org_name']}")
            print(f"  True Label: {fn_example['true_label']} (Successful)")
            print(f"  Predicted: {fn_example['predicted_raw']} (Interpreted as Unsuccessful)")
            print(f"  Analysis Snippet: {fn_example['analysis']}")
        else:
            print("\\nNo False Negative example found.")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred processing {file_path} with model {model_type}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_fault_examples.py <file_path> <model_type>", file=sys.stderr)
        print("model_type can be 'vanilla', 'foundergpt', 'cot', 'ssff_basic', 'ssff_pro', 'ssff_nl_basic', 'ssff_nl_pro', or 'raise'", file=sys.stderr)
        sys.exit(1)

    file_path_arg = sys.argv[1]
    model_type_arg = sys.argv[2]

    if model_type_arg not in ['vanilla', 'foundergpt', 'cot', 'ssff_basic', 'ssff_pro', 'ssff_nl_basic', 'ssff_nl_pro', 'raise']:
        print(f"Invalid model_type: {model_type_arg}. Must be one of 'vanilla', 'foundergpt', 'cot', 'ssff_basic', 'ssff_pro', 'ssff_nl_basic', 'ssff_nl_pro', 'raise'", file=sys.stderr)
        sys.exit(1)
        
    print(f"Processing {model_type_arg.upper()} from {file_path_arg}...")
    find_misclassified_examples(file_path_arg, model_type_arg) 
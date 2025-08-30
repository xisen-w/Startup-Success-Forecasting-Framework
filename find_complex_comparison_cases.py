import json
import sys

def load_data_to_map(file_path, model_name_for_error_msg=""):
    """Loads JSON data from a file into a dictionary keyed by org_uuid."""
    data_map = {}
    raw_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                raw_data = json.load(f)
                if not isinstance(raw_data, list):
                    print(f"Error: Expected a list of JSON objects in {file_path} for {model_name_for_error_msg}, but got {type(raw_data)}. Trying line-by-line.", file=sys.stderr)
                    raw_data = []
                    f.seek(0)
            except json.JSONDecodeError:
                print(f"Initial JSON load failed for {file_path} ({model_name_for_error_msg}). Attempting to read line-by-line.", file=sys.stderr)
                f.seek(0)
                raw_data = []

            if not raw_data: # If initial load failed or was not a list
                print(f"Processing {file_path} ({model_name_for_error_msg}) as newline-delimited JSON objects.", file=sys.stderr)
                f.seek(0)
                temp_data_list = []
                for line_number, line in enumerate(f):
                    line = line.strip()
                    if line.startswith('[') and line_number == 0: line = line[1:]
                    if line.endswith(']') and not line.startswith('['): line = line[:-1]
                    if line.endswith(','): line = line[:-1]
                    if line:
                        try:
                            temp_data_list.append(json.loads(line))
                        except json.JSONDecodeError as e_line:
                            # print(f"Warning: Could not decode JSON object on line {line_number + 1} in {file_path}. Error: {e_line}. Line: '{line[:100]}...'", file=sys.stderr)
                            pass # Be more silent for this complex script unless it fails entirely
                raw_data = temp_data_list
        
        if not raw_data:
            print(f"Error: No data could be loaded from {file_path} for {model_name_for_error_msg}.", file=sys.stderr)
            return None

        for record in raw_data:
            if isinstance(record, dict) and "org_uuid" in record:
                data_map[record["org_uuid"]] = record
            # else:
                # print(f"Warning: Record in {file_path} missing 'org_uuid' or not a dict: {str(record)[:100]}...", file=sys.stderr)
        return data_map
    except FileNotFoundError:
        print(f"Error: File not found at {file_path} ({model_name_for_error_msg})", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading {file_path} ({model_name_for_error_msg}): {e}", file=sys.stderr)
        return None

def get_binary_prediction(model_name, record):
    """Extracts and normalizes prediction to 0 or 1."""
    true_label = record.get("label")
    
    pred_raw = None
    if model_name == "baseline":
        pred_raw = record.get("recommendation")
    elif model_name == "cot":
        pred_raw = record.get("recommendation")
    elif model_name == "ssff_basic":
        analysis = record.get("basic_analysis_structured")
        if analysis and isinstance(analysis, dict):
            pred_raw = analysis.get("outcome")
    elif model_name == "ssff_pro":
        analysis = record.get("final_analysis_pro_structured")
        if analysis and isinstance(analysis, dict):
            pred_raw = analysis.get("outcome")

    if pred_raw is None:
        return None, true_label is None # prediction, label_missing

    if model_name in ["baseline", "cot"]:
        if pred_raw == "Successful": return 1, true_label is None
        if pred_raw == "Unsuccessful": return 0, true_label is None
    elif model_name in ["ssff_basic", "ssff_pro"]:
        if pred_raw == "Invest": return 1, true_label is None
        if pred_raw in ["Hold", "Pass", "Unsuccessful"]: return 0, true_label is None
    
    return None, true_label is None # Unknown prediction string

def main(baseline_file, cot_file, ssff_file):
    baseline_map = load_data_to_map(baseline_file, "Baseline")
    cot_map = load_data_to_map(cot_file, "CoT")
    ssff_map = load_data_to_map(ssff_file, "SSFF Regular")

    if not all([baseline_map, cot_map, ssff_map]):
        print("Error: One or more data files could not be loaded. Exiting.", file=sys.stderr)
        return

    found_case1 = False
    found_case2 = False

    # Iterate through org_uuids that are common to all three maps
    common_uuids = set(baseline_map.keys()) & set(cot_map.keys()) & set(ssff_map.keys())
    if not common_uuids:
        print("No common org_uuids found across all files.", file=sys.stderr)
        return

    print(f"Processing {len(common_uuids)} common companies...")

    for org_uuid in common_uuids:
        if found_case1 and found_case2:
            break

        baseline_record = baseline_map[org_uuid]
        cot_record = cot_map[org_uuid]
        ssff_record = ssff_map[org_uuid]

        org_name = ssff_record.get("org_name", baseline_record.get("org_name", cot_record.get("org_name", "N/A")))
        true_label = ssff_record.get("label") # Assume label is consistent, take from ssff

        if true_label is None: # If any primary record lacks a label, skip
            # Fallback to check other records for label, though ideally it's in ssff_record
            if baseline_record.get("label") is not None: true_label = baseline_record.get("label")
            elif cot_record.get("label") is not None: true_label = cot_record.get("label")
            else:
                # print(f"Skipping {org_name} ({org_uuid}) due to missing true label.", file=sys.stderr)
                continue
        
        pred_baseline, lbl_missing_baseline = get_binary_prediction("baseline", baseline_record)
        pred_cot, lbl_missing_cot = get_binary_prediction("cot", cot_record)
        pred_basic, lbl_missing_basic = get_binary_prediction("ssff_basic", ssff_record)
        pred_pro, lbl_missing_pro = get_binary_prediction("ssff_pro", ssff_record)

        # If any prediction could not be determined, skip this record for safety
        if None in [pred_baseline, pred_cot, pred_basic, pred_pro]:
            # print(f"Skipping {org_name} ({org_uuid}) due to an undetermined prediction for one of the models.", file=sys.stderr)
            continue

        # Case 1: Baseline WRONG, CoT WRONG, Basic CORRECT, Pro CORRECT
        if not found_case1:
            is_baseline_wrong = (pred_baseline != true_label)
            is_cot_wrong = (pred_cot != true_label)
            is_basic_correct = (pred_basic == true_label)
            is_pro_correct = (pred_pro == true_label)

            if is_baseline_wrong and is_cot_wrong and is_basic_correct and is_pro_correct:
                print("\n--- CASE 1 FOUND ---")
                print(f"Organization: {org_name} (UUID: {org_uuid})")
                print(f"True Label: {true_label} ({'Successful' if true_label == 1 else 'Unsuccessful'})")
                print(f"  Baseline Prediction: {pred_baseline} (Correct: {not is_baseline_wrong})")
                print(f"  CoT Prediction:      {pred_cot} (Correct: {not is_cot_wrong})")
                print(f"  SSFF Basic Pred:   {pred_basic} (Correct: {is_basic_correct})")
                print(f"  SSFF Pro Pred:     {pred_pro} (Correct: {is_pro_correct})")
                found_case1 = True

        # Case 2: Baseline WRONG, CoT WRONG, Basic WRONG, Pro CORRECT
        if not found_case2:
            is_baseline_wrong = (pred_baseline != true_label)
            is_cot_wrong = (pred_cot != true_label)
            is_basic_wrong = (pred_basic != true_label)
            is_pro_correct = (pred_pro == true_label)

            if is_baseline_wrong and is_cot_wrong and is_basic_wrong and is_pro_correct:
                print("\n--- CASE 2 FOUND ---")
                print(f"Organization: {org_name} (UUID: {org_uuid})")
                print(f"True Label: {true_label} ({'Successful' if true_label == 1 else 'Unsuccessful'})")
                print(f"  Baseline Prediction: {pred_baseline} (Correct: {not is_baseline_wrong})")
                print(f"  CoT Prediction:      {pred_cot} (Correct: {not is_cot_wrong})")
                print(f"  SSFF Basic Pred:   {pred_basic} (Correct: {not is_basic_wrong})")
                print(f"  SSFF Pro Pred:     {pred_pro} (Correct: {is_pro_correct})")
                found_case2 = True
                
    if not found_case1:
        print("\nNo example found for Case 1.")
    if not found_case2:
        print("\nNo example found for Case 2.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python find_complex_comparison_cases.py <baseline_json_path> <cot_json_path> <ssff_regular_json_path>", file=sys.stderr)
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2], sys.argv[3]) 
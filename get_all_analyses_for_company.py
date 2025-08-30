import json
import sys

def load_record_by_uuid(file_path, org_uuid, model_name_for_error_msg=""):
    """Loads a specific record from a JSON file by org_uuid."""
    raw_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                raw_data = json.load(f)
                if not isinstance(raw_data, list):
                    print(f"Error: Expected a list of JSON objects in {file_path} for {model_name_for_error_msg}. Trying line-by-line.", file=sys.stderr)
                    raw_data = []
                    f.seek(0)
            except json.JSONDecodeError:
                print(f"Initial JSON load failed for {file_path} ({model_name_for_error_msg}). Attempting to read line-by-line.", file=sys.stderr)
                f.seek(0)
                raw_data = []

            if not raw_data:
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
                        except json.JSONDecodeError:
                            pass 
                raw_data = temp_data_list
        
        if not raw_data:
            print(f"Error: No data could be loaded from {file_path} for {model_name_for_error_msg}.", file=sys.stderr)
            return None

        for record in raw_data:
            if isinstance(record, dict) and record.get("org_uuid") == org_uuid:
                return record
        print(f"Record with org_uuid {org_uuid} not found in {file_path}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"Error: File not found at {file_path} ({model_name_for_error_msg})", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading {file_path} ({model_name_for_error_msg}): {e}", file=sys.stderr)
        return None

def print_analysis_details(model_name, record):
    org_name = record.get("org_name", "N/A")
    true_label = record.get("label")
    true_label_str = f"{'Successful' if true_label == 1 else 'Unsuccessful'}" if true_label is not None else "Unknown"
    
    pred_raw = "N/A"
    analysis_text = "N/A"
    score = "N/A"

    print(f"\n--- {model_name.upper()} Analysis for: {org_name} (True Label: {true_label_str}) ---")

    if model_name == "baseline":
        pred_raw = record.get("recommendation", "N/A")
        analysis_text = record.get("total_analysis", "No analysis text.")
        score = record.get("score", "N/A")
    elif model_name == "cot":
        pred_raw = record.get("recommendation", "N/A")
        analysis_text = record.get("total_analysis", "No analysis text.") # CoT also uses total_analysis
        score = record.get("score", "N/A")
    elif model_name == "ssff_basic":
        basic_analysis = record.get("basic_analysis_structured")
        if basic_analysis and isinstance(basic_analysis, dict):
            pred_raw = basic_analysis.get("outcome", "N/A")
            analysis_text = basic_analysis.get("IntegratedAnalysis", "No analysis text.")
            score = basic_analysis.get("overall_score", "N/A")
    elif model_name == "ssff_pro":
        pro_analysis = record.get("final_analysis_pro_structured")
        if pro_analysis and isinstance(pro_analysis, dict):
            pred_raw = pro_analysis.get("outcome", "N/A")
            analysis_text = pro_analysis.get("IntegratedAnalysis", "No analysis text.")
            score = pro_analysis.get("overall_score", "N/A")
    else:
        print("Unknown model type for printing.")
        return

    print(f"  Raw Prediction: {pred_raw}")
    print(f"  Score: {score}")
    print(f"  Analysis Text Snippet (up to 1000 chars):\n    {analysis_text[:1000].strip()}\n    ...")

def main(org_uuid_to_find, baseline_file, cot_file, ssff_file):
    print(f"Attempting to retrieve analyses for company UUID: {org_uuid_to_find}")

    baseline_record = load_record_by_uuid(baseline_file, org_uuid_to_find, "Baseline")
    cot_record = load_record_by_uuid(cot_file, org_uuid_to_find, "CoT")
    ssff_record = load_record_by_uuid(ssff_file, org_uuid_to_find, "SSFF Regular")

    if baseline_record:
        print_analysis_details("baseline", baseline_record)
    else:
        print(f"\nBaseline record not found for {org_uuid_to_find}.")

    if cot_record:
        print_analysis_details("cot", cot_record)
    else:
        print(f"\nCoT record not found for {org_uuid_to_find}.")

    if ssff_record:
        print_analysis_details("ssff_basic", ssff_record)
        print_analysis_details("ssff_pro", ssff_record) # Both basic and pro are in the same ssff_record
    else:
        print(f"\nSSFF Regular record not found for {org_uuid_to_find} (needed for Basic and Pro).")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python get_all_analyses_for_company.py <org_uuid> <baseline_json_path> <cot_json_path> <ssff_regular_json_path>", file=sys.stderr)
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]) 
import json
import sys

def find_company_analysis(file_path, model_type, target_org_name):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Simplified loading assuming a list of JSON objects for this one-off script
            data = json.load(f)
            if not isinstance(data, list):
                print(f"Error: Expected a list of JSON objects in {file_path}", file=sys.stderr)
                # Attempt line-by-line
                f.seek(0)
                data = []
                for line_number, line in enumerate(f):
                    line = line.strip()
                    if line.startswith('[') and line_number == 0: line = line[1:]
                    if line.endswith(']') and not line.startswith('['): line = line[:-1]
                    if line.endswith(','): line = line[:-1]
                    if line:
                        try: data.append(json.loads(line))
                        except json.JSONDecodeError: continue
                if not data: return

        for record in data:
            if not isinstance(record, dict):
                continue
            
            org_name = record.get("org_name")
            if org_name == target_org_name:
                true_label = record.get("label")
                predicted_outcome_raw = None
                analysis_text = "N/A"

                if model_type == 'ssff_basic':
                    basic_analysis = record.get("basic_analysis_structured")
                    if basic_analysis and isinstance(basic_analysis, dict):
                        predicted_outcome_raw = basic_analysis.get("outcome")
                        analysis_text = basic_analysis.get("IntegratedAnalysis", "N/A")
                elif model_type == 'ssff_pro':
                    pro_analysis = record.get("final_analysis_pro_structured")
                    if pro_analysis and isinstance(pro_analysis, dict):
                        predicted_outcome_raw = pro_analysis.get("outcome")
                        analysis_text = pro_analysis.get("IntegratedAnalysis", "N/A")
                
                print(f"--- Analysis for {target_org_name} using {model_type.upper()} ---")
                print(f"  True Label: {true_label}")
                print(f"  Predicted Outcome: {predicted_outcome_raw}")
                print(f"  Integrated Analysis Snippet: {analysis_text[:1000]}") # Increased snippet length
                return

        print(f"{target_org_name} not found in {file_path}", file=sys.stderr)

    except Exception as e:
        print(f"Error processing {file_path} for {target_org_name} ({model_type}): {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <file_path> <model_type: ssff_basic|ssff_pro> <target_org_name>", file=sys.stderr)
        sys.exit(1)
    find_company_analysis(sys.argv[1], sys.argv[2], sys.argv[3]) 
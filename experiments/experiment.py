import sys
import os
import pandas as pd
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import json

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from ssff_framework import StartupFramework

class AnalystTeam(Enum):
    ALL_AGENTS = "All-Agents"
    ONE_AGENT = "Only One Agent"
    ZERO_SHOT = "Zero-Shot"

class InputType(Enum):
    FULL_INFO = "Full Info"
    LIMITED_INFO = "Limited Info"

class Method(Enum):
    SSFF_SETUP = "SSFF_Setup"
    BASELINE = "Baseline"

@dataclass
class ExperimentSetup:
    input_type: InputType
    method: Method
    analysis: str
    final_decision: bool
    fifs: Optional[bool] = None
    random_forest_classification: Optional[bool] = None
    quant_evaluation: Optional[bool] = None
    founder_level_segmentation: Optional[bool] = None
    analyst_team: Optional[AnalystTeam] = None
    if_external: Optional[bool] = None

    def to_json(self):
        setup_dict = asdict(self)
        setup_dict['input_type'] = self.input_type.value
        setup_dict['method'] = self.method.value
        if self.analyst_team:
            setup_dict['analyst_team'] = self.analyst_team.value
        return json.dumps(setup_dict)

    @classmethod
    def from_json(cls, json_str):
        setup_dict = json.loads(json_str)
        setup_dict['input_type'] = InputType(setup_dict['input_type'])
        setup_dict['method'] = Method(setup_dict['method'])
        if setup_dict.get('analyst_team'):
            setup_dict['analyst_team'] = AnalystTeam(setup_dict['analyst_team'])
        return cls(**setup_dict)

@dataclass
class Company:
    name: str
    startup_desc: str
    founder_desc: str
    short_desc: str

@dataclass
class ExperimentResult:
    name: str
    company: str
    setup: ExperimentSetup
    prediction: str
    analysis: Dict[str, Any]
    final_decision: Optional[str] = None
    founder_idea_fit: Optional[float] = None
    random_forest_prediction: Optional[str] = None
    quantitative_evaluation: Optional[float] = None
    founder_segmentation: Optional[str] = None

    def to_json(self):
        result_dict = asdict(self)
        result_dict['setup'] = self.setup.to_json()
        return json.dumps(result_dict)

    @classmethod
    def from_json(cls, json_str):
        result_dict = json.loads(json_str)
        result_dict['setup'] = ExperimentSetup.from_json(result_dict['setup'])
        return cls(**result_dict)

class Experiment:
    def __init__(self, name: str, setup: ExperimentSetup):
        self.name = name
        self.setup = setup
        self.framework = StartupFramework()

    def run(self, company: Company) -> ExperimentResult:
        startup_info_str = f"""
        Company Name: {company.name}
        Description: {company.startup_desc}
        Founder Information: {company.founder_desc}
        """

        if self.setup.method == Method.BASELINE:
            if "GPT" in self.name:
                result = self.framework.baseline_gpt_analysis(startup_info_str)
            else:  # ML Baseline
                result = self.framework.baseline_ml_analysis(startup_info_str)
        else:  # SSFF_SETUP
            result = self.framework.analyze_startup(startup_info_str)

        # Configure the analysis based on the experiment setup
        if not self.setup.final_decision:
            result.pop('Final Decision', None)
        if not self.setup.fifs:
            result.pop('Founder Idea Fit', None)
        if not self.setup.random_forest_classification:
            result.pop('Random Forest Prediction', None)
        if not self.setup.quant_evaluation:
            result.pop('Quantitative Decision', None)
        if not self.setup.founder_level_segmentation:
            result.pop('Founder Segmentation', None)

        result = {
            "name": self.name,
            "company": company.name,
            "setup": self.setup,
            "prediction": result.get('Categorical Prediction', 'N/A'),
            "analysis": result,
            "final_decision": result.get('Final Decision'),
            "founder_idea_fit": result.get('Founder Idea Fit'),
            "random_forest_prediction": result.get('Random Forest Prediction'),
            "quantitative_evaluation": result.get('Quantitative Decision'),
            "founder_segmentation": result.get('Founder Segmentation')
        }

        return ExperimentResult(**result)

def load_dataset(file_path: str) -> Company:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file {file_path} does not exist.")

    try:
        df = pd.read_csv(file_path)
        first_row = df.iloc[0]
        
        return Company(
            name=first_row['org_name'],
            startup_desc=first_row['long_description'],
            founder_desc=first_row['paragraph'],
            short_desc=first_row['short_description']
        )
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {str(e)}")

def run_experiments(company: Company):
    experiments = [
        Experiment("Baseline GPT", ExperimentSetup(InputType.FULL_INFO, Method.BASELINE, "GPT Analysis", True)),
        Experiment("Baseline ML", ExperimentSetup(InputType.FULL_INFO, Method.BASELINE, "ML-Baseline Prediction", True)),
        Experiment("SSFF Basic", ExperimentSetup(InputType.FULL_INFO, Method.SSFF_SETUP, "SSFF Analyst Block Only", True, analyst_team=AnalystTeam.ALL_AGENTS)),
        Experiment("SSFF Pro", ExperimentSetup(InputType.FULL_INFO, Method.SSFF_SETUP, "SSFF Analyst Block + External Block", True, if_external=True, analyst_team=AnalystTeam.ALL_AGENTS)),
        Experiment("SSFF Max", ExperimentSetup(InputType.FULL_INFO, Method.SSFF_SETUP, "SSFF Max", True, True, True, True, True, AnalystTeam.ALL_AGENTS, True)),
        Experiment("Ablation 1", ExperimentSetup(InputType.FULL_INFO, Method.SSFF_SETUP, "SSFF Max Without Founder Level Segmentation", True, True, True, True, False, AnalystTeam.ALL_AGENTS, True)),
        Experiment("Ablation 2", ExperimentSetup(InputType.FULL_INFO, Method.SSFF_SETUP, "SSFF Max Without FIFS", True, False, True, True, True, AnalystTeam.ALL_AGENTS, True)),
        Experiment("Ablation 3", ExperimentSetup(InputType.FULL_INFO, Method.SSFF_SETUP, "SSFF Max Without Random Forest Classification", True, True, False, True, True, AnalystTeam.ALL_AGENTS, True)),
        Experiment("Ablation 4", ExperimentSetup(InputType.FULL_INFO, Method.SSFF_SETUP, "SSFF Max Without Quantitative Evaluation", True, True, True, False, True, AnalystTeam.ALL_AGENTS, True)),
        Experiment("Ablation 5", ExperimentSetup(InputType.FULL_INFO, Method.SSFF_SETUP, "SSFF Max Without Product Agent", True, True, True, True, True, AnalystTeam.ONE_AGENT, True)),
        Experiment("Ablation 6", ExperimentSetup(InputType.FULL_INFO, Method.SSFF_SETUP, "SSFF Max Without Market Agent", True, True, True, True, True, AnalystTeam.ONE_AGENT, True)),
        Experiment("Ablation 7", ExperimentSetup(InputType.FULL_INFO, Method.SSFF_SETUP, "SSFF Max Without Founder Agent", True, True, True, True, True, AnalystTeam.ONE_AGENT, True)),
    ]

    results = []
    for experiment in experiments:
        try:
            result = experiment.run(company)
            results.append(result)
            print(f"Completed experiment: {experiment.name}")
        except Exception as e:
            print(f"An error occurred in experiment {experiment.name}: {str(e)}")

    return results

def save_results(results: List[ExperimentResult], output_file: str):
    flattened_results = []
    for result in results:
        flat_result = asdict(result)
        flat_result['setup'] = result.setup.to_json()
        flattened_results.append(flat_result)

    df = pd.DataFrame(flattened_results)
    df.to_csv(output_file, index=False)

def load_results(input_file: str) -> List[ExperimentResult]:
    df = pd.read_csv(input_file)
    results = []
    for _, row in df.iterrows():
        result_dict = row.to_dict()
        result_dict['setup'] = ExperimentSetup.from_json(result_dict['setup'])
        results.append(ExperimentResult(**result_dict))
    return results

def main():
    input_file = os.path.join(project_root, 'data', 'Merged_Successful_V2')
    output_file = 'results.csv'

    try:
        company = load_dataset(input_file)
        print(f"Loaded company: {company.name}")
        print(f"Startup description: {company.startup_desc[:100]}...")
        print(f"Founder description: {company.founder_desc[:100]}...")
        print(f"Short description: {company.short_desc}")

        results = run_experiments(company)
        save_results(results, output_file)
        
        print(f"\nExperiments completed. Results saved to {output_file}")
        
        # Load and print results
        loaded_results = load_results(output_file)
        for result in loaded_results:
            print(f"\n{result['name']}:")
            print(f"Setup: {result['setup']}")
            print(f"Prediction: {result['prediction']}")
            print("Analysis Keys:", ', '.join(result['analysis'].keys()))
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

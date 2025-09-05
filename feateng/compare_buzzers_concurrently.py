import itertools
import os
import subprocess
import sys
import pandas as pd
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

LOSS_FUNCTIONS = {
    "MLP": "BuzzLoss",
    "LogisticBuzzer": "Logistic Loss",
}

features = ["Length", "Frequency", "Category", "ContextualMatch", "PreviousGuess"]

results_df = pd.DataFrame(columns=[
    "Features", "Buzzer Type", "Filename Stem", "Loss Function", "Training Limit", "Testing Limit",
    "Training Dataset", "Test Dataset", "Evaluation",
    "best %", "timid %", "hit %", "close %", "miss %", "aggressive %", "waiting %",
    "Questions Right", "Total", "Accuracy", "Buzz Ratio", "Buzz Position"
])

def generate_filename_stem(subset, buzzer_type="LogisticBuzzer"):
    buzzer_str = "logit" if buzzer_type == "LogisticBuzzer" else buzzer_type.lower()
    if not subset:
        return f"{buzzer_str}_no_features"
    elif set(subset) == set(features):
        return f"{buzzer_str}_with_all_features"
    else:
        return f"{buzzer_str}_with_" + "_".join(subset).lower()

def validate_json_output(json_path):
    try:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Output JSON file not found: {json_path}")
        if os.path.getsize(json_path) == 0:
            raise ValueError(f"Output JSON file is empty: {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)
            if not data:
                raise ValueError(f"Output JSON file contains invalid or empty data: {json_path}")
        return data
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        return str(e)

def process_subset(buzzer_type, subset):
    try:
        filename_stem = generate_filename_stem(subset, buzzer_type)
        training_limit, testing_limit = 50, 25
        training_dataset = "../data/qanta.buzztrain.json.gz"
        test_dataset = "../data/qanta.buzzdev.json.gz"
        guesser_model_train = "../models/buzztrain_gpt4o_cache"
        guesser_model_test = "../models/buzzdev_gpt4o_cache"

        buzzer_command = [
            sys.executable, 'buzzer.py', '--guesser_type=Gpr', '--limit', str(training_limit),
            '--GprGuesser_filename', guesser_model_train, '--questions', training_dataset,
            '--buzzer_guessers', 'Gpr', '--buzzer_type', buzzer_type
        ]
        if subset:
            buzzer_command.extend(['--features'] + list(subset))
        buzzer_command.append(f'--{buzzer_type}Buzzer_filename=models/{filename_stem}')

        output_json = f"summary/eval_output_{filename_stem}.json"
        eval_command = [
            sys.executable, 'eval.py', '--guesser_type=Gpr', '--limit', str(testing_limit),
            '--questions', test_dataset, '--buzzer_guessers', 'Gpr',
            '--GprGuesser_filename', guesser_model_test, '--evaluate', "buzzer",
            '--output_json', output_json
        ]
        eval_command.append(f'--{buzzer_type}Buzzer_filename=models/{filename_stem}')
        if subset:
            eval_command.extend(['--features'] + list(subset))

        subprocess.run(buzzer_command, check=True)
        subprocess.run(eval_command, check=True)

        validation_result = validate_json_output(output_json)
        if isinstance(validation_result, dict):
            eval_results = validation_result
        else:
            raise ValueError(f"Validation failed: {validation_result}")

        loss_function = LOSS_FUNCTIONS.get(buzzer_type, "Unknown")
        new_row = {
            "Features": list(subset), "Buzzer Type": buzzer_type, "Filename Stem": filename_stem,
            "Loss Function": loss_function, "Training Limit": training_limit,
            "Testing Limit": testing_limit, "Training Dataset": training_dataset,
            "Test Dataset": test_dataset, "Evaluation": "buzzer",
            **eval_results["outcome_percentages"], "Questions Right": eval_results["questions_right"],
            "Total": eval_results["total"], "Accuracy": eval_results["accuracy"],
            "Buzz Ratio": eval_results["buzz_ratio"], "Buzz Position": eval_results["buzz_position"]
        }
        return new_row
    except Exception as e:
        print(f"Error processing subset {subset} for {buzzer_type}: {e}")
        return None

buzzer_models = ["MLP", "LogisticBuzzer"]
feature_subsets = list(itertools.chain.from_iterable(itertools.combinations(features, r) for r in range(len(features) + 1)))
feature_subsets = [["Length", "Frequency", "Category", "ContextualMatch", "PreviousGuess"]]

with ThreadPoolExecutor() as executor:
    futures = []
    for buzzer_type in buzzer_models:
        for subset in feature_subsets:
            futures.append(executor.submit(process_subset, buzzer_type, subset))
    results = [future.result() for future in futures if future.result() is not None]

if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv("summary/compare_buzzers_concurrently_eval_summary.csv", index=False)
else:
    print("No results generated.")

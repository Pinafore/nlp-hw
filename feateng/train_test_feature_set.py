import itertools
import os
import subprocess
import sys
import pandas as pd
import json

# Define the features to use in generating the power set
features = ["Length", "Frequency", "Category", "ContextualMatch", "PreviousGuess"]

# DataFrame to store results
results_df = pd.DataFrame(columns=[
    "Features", "Filename Stem", "Training Limit", "Testing Limit",
    "Training Dataset", "Test Dataset", "Evaluation",
    "best %", "timid %", "hit %", "close %", "miss %", "aggressive %", "waiting %",
    "Questions Right", "Total", "Accuracy", "Buzz Ratio", "Buzz Position"
])

# Function to generate the filename stem based on the subset of features
def generate_filename_stem(subset):
    if not subset:
        return "no_features"
    elif set(subset) == set(features):
        return "with_all_features"
    else:
        return "with_" + "_".join(subset).lower()

# Generate the power set of features
feature_subsets = list(itertools.chain.from_iterable(itertools.combinations(features, r) for r in range(len(features)+1)))

# Loop over each subset of features to construct and execute the commands
for subset in feature_subsets:
    # Determine the filename stem based on the subset
    filename_stem = generate_filename_stem(subset)
    
    # Set values for the parameters
    training_limit = 50
    testing_limit = 25
    training_dataset = "../data/qanta.buzztrain.json.gz"
    test_dataset = "../data/qanta.buzzdev.json.gz"
    evaluation = "buzzer"
    
    # Construct the `buzzer.py` command using sys.executable
    buzzer_command = [
        sys.executable, 'buzzer.py', '--guesser_type=Gpr', '--limit', str(training_limit),
        '--GprGuesser_filename=../models/buzztrain_gpr_cache',
        '--questions', training_dataset, '--buzzer_guessers', 'Gpr'
    ]
    # Only add --features if subset is not empty
    if subset:
        buzzer_command.extend(['--features'] + list(subset))
    
    buzzer_command.extend(['--LogisticBuzzer_filename=models/' + filename_stem])
    
    # Construct the `eval.py` command using sys.executable
    eval_command = [
        sys.executable, 'eval.py', '--guesser_type=Gpr',
        '--TfidfGuesser_filename=models/TfidfGuesser', '--limit', str(testing_limit),
        '--questions', test_dataset, '--buzzer_guessers', 'Gpr',
        '--GprGuesser_filename=../models/buzzdev_gpr_cache',
        '--LogisticBuzzer_filename=models/' + filename_stem,
        '--evaluate', evaluation
    ]
    # Only add --features if subset is not empty
    if subset:
        eval_command.extend(['--features'] + list(subset))
    
    # Execute the commands
    print(f"Running with feature subset: {subset} -> {filename_stem}")
    
    # Run the buzzer.py command
    subprocess.run(buzzer_command, check=True)
    
    # Run the eval.py command
    subprocess.run(eval_command, check=True)

    # Load results from JSON file
    with open("summary/eval_output.json", "r") as f:
        eval_results = json.load(f)
    
    # Create a DataFrame for the new row
    new_row_df = pd.DataFrame([{
        "Features": list(subset),
        "Filename Stem": filename_stem,
        "Training Limit": training_limit,
        "Testing Limit": testing_limit,
        "Training Dataset": training_dataset,
        "Test Dataset": test_dataset,
        "Evaluation": evaluation,
        **eval_results["outcome_percentages"],
        "Questions Right": eval_results["questions_right"],
        "Total": eval_results["total"],
        "Accuracy": eval_results["accuracy"],
        "Buzz Ratio": eval_results["buzz_ratio"],
        "Buzz Position": eval_results["buzz_position"]
    }])

    # Use pd.concat to add the new row to results_df
    results_df = pd.concat([results_df, new_row_df], ignore_index=True)

# Sort the DataFrame by descending order of Buzz Ratio
results_df = results_df.sort_values(by="Buzz Ratio", ascending=False)

# Export the DataFrame as CSV
os.makedirs("summary", exist_ok=True)
results_df.to_csv("summary/eval_summary.csv", index=False)

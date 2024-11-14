import itertools
import os
import subprocess

# Define the features to use in generating the power set
features = ["Length", "Frequency", "ContextualMatch", "Category", "PreviousGuess"]

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
    
    # Join the subset of features for CLI input
    features_str = " ".join(subset)
    
    # Construct the `buzzer.py` command
    buzzer_command = f"""
    python3 buzzer.py --guesser_type=Gpr --limit=500 \
    --GprGuesser_filename=../models/buzztrain_gpr_cache \
    --questions=../data/qanta.buzztrain.json.gz --buzzer_guessers Gpr \
    --features {features_str} \
    --LogisticBuzzer_filename=models/{filename_stem}
    """
    
    # Construct the `eval.py` command with redirection to the output .txt file
    eval_command = f"""
    python3 eval.py --guesser_type=Gpr \
    --TfidfGuesser_filename=models/TfidfGuesser --limit=25 \
    --questions=../data/qanta.buzzdev.json.gz --buzzer_guessers Gpr \
    --GprGuesser_filename=../models/buzzdev_gpr_cache \
    --LogisticBuzzer_filename=models/{filename_stem} --features {features_str} \
    > evals/eval_output_{filename_stem}.txt
    """
    
    # Execute the commands
    print(f"Running with feature subset: {subset} -> {filename_stem}")
    
    # Run the buzzer.py command
    subprocess.run(buzzer_command, shell=True, check=True)
    
    # Run the eval.py command
    subprocess.run(eval_command, shell=True, check=True)

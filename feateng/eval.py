# Jordan Boyd-Graber
# 2023
#
# Run an evaluation on a QA system and print results

from random import random

from buzzer import rough_compare

from params import load_guesser, load_questions, load_buzzer, \
    add_buzzer_params, add_guesser_params, add_general_params,\
    add_question_params, setup_logging

kLABELS = {"best": "Guess was correct, Buzzer correctly detected this (true positive)",
           "timid": "Guess was correct, Buzzer did not recognize this (false negative)",
           "aggressive": "Guess was wrong, Buzzer thought it was correct (false positive)",
           "waiting": "Guess was wrong, Buzzer correctly detected this (true negative)"}

def eval_retrieval(guesser, questions, num_test, n_guesses=1, cutoff=None):
    """
    Evaluate the guesser's retrieval
    """
    
    recall = 0
    correct = 0
    total = 0
    for question in questions:
        text = question["text"]
        if cutoff is None:
            text = text[:int(random() * len(text))]
        else:
            text = text[:cutoff]

        guesses = guesser(text)
        answer = question["page"]

        print(answer, guesses)            
        
        if answer in [x["guess"] for x in guesses]:
            recall += 1
        if rough_compare(guesses[0]["guess"], answer):
            correct += 1
        total += 1

        if total >= num_test:
            break

    return correct, recall, total

def pretty_feature_print(features, first_features=["guess", "answer", "id"]):
    """
    Nicely print a buzzer example's features
    """
    
    import textwrap
    wrapper = textwrap.TextWrapper()

    lines = []

    for ii in first_features:
        lines.append("%20s: %s" % (ii, features[ii]))
    for ii in [x for x in features if x not in first_features]:
        if isinstance(features[ii], str):
            if len(features[ii]) > 70:
                long_line = "%20s: %s" % (ii, "\n                      ".join(wrapper.wrap(features[ii])))
                lines.append(long_line)
            else:
                lines.append("%20s: %s" % (ii, features[ii]))
        elif isinstance(features[ii], float):
            lines.append("%20s: %0.4f" % (ii, features[ii]))
        else:
            lines.append("%20s: %s" % (ii, str(features[ii])))
    lines.append("--------------------")
    return "\n".join(lines)


def eval_buzzer(buzzer, questions):
    """
    Compute buzzer outcomes on a dataset
    """
    
    from collections import Counter, defaultdict
    
    buzzer.load()
    buzzer.add_data(questions)
    predict, feature_matrix, feature_dict, correct, metadata = buzzer.predict(questions)
    outcomes = Counter()
    examples = defaultdict(list)
    for buzz, guess_correct, features, meta in zip(predict, correct, feature_dict, metadata):
        # Add back in metadata now that we have prevented cheating in feature creation
        for ii in meta:
            features[ii] = meta[ii]
        if guess_correct:
            if buzz:
                outcomes["best"] += 1
                examples["best"].append(features)
            else:
                outcomes["timid"] += 1
                examples["timid"].append(features)
        else:
            if buzz:
                outcomes["aggressive"] += 1
                examples["aggressive"].append(features)
            else:
                outcomes["waiting"] += 1
                examples["waiting"].append(features)
    return outcomes, examples
                

if __name__ == "__main__":
    # Load model and evaluate it
    import argparse
    import random
    
    parser = argparse.ArgumentParser()
    add_general_params(parser)
    add_guesser_params(parser)
    add_question_params(parser)
    add_buzzer_params(parser)

    parser.add_argument('--evaluate', default="buzzer", type=str)
    
    flags = parser.parse_args()
    setup_logging(flags)

    questions = load_questions(flags)    
    if flags.evaluate == "buzzer":
        buzzer = load_buzzer(flags)
        guesser = load_guesser(flags)
        outcomes, examples = eval_buzzer(buzzer, questions)
        total = sum(outcomes.values())
        for ii in outcomes:
            print("%s %0.2f\n===================\n" % (ii, outcomes[ii] / total))
            if len(examples[ii]) > 10:
                population = list(random.sample(examples[ii], 10))
            else:
                population = examples[ii]
            for jj in population:
                print(pretty_feature_print(jj))
            print("=================")
        for weight, feature in zip(buzzer._classifier.coef_[0], buzzer._featurizer.feature_names_):
            print("%40s: %0.4f" % (feature.strip(), weight))
        for ii in outcomes:
            print("%20s %0.2f\n" % (ii, outcomes[ii] / total))
        print("Accuracy: %0.2f  Buzz ratio: %0.2f" %
              ((outcomes["best"] + outcomes["waiting"]) / total,
               outcomes["best"] - outcomes["aggressive"] * 0.5))
    elif flags.evaluate == "guesser":
        print(eval_retrieval(guesser, questions, 100))

    

from functools import partial
from collections import defaultdict
import logging

from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from buzzer import BuzzerParameters

from buzzer import Buzzer

class LoraBertParameters(BuzzerParameters):
    def __init__(self, customized_params=None):
        BuzzerParameters.__init__(self)
        self.name = "lorabert_buzzer"
        if customized_params:
            self.params += customized_params
        else:
            lorabert_params = [("rank", int, 16, "Rank of LoRA adaptation"),
                               ("base_model", str, "distilbert-base-uncased", "The HF model we will adapt"),
                               ("alpha", float, 1.0, "")]
            self.params += lorabert_params
                               

    # TODO: These should be inherited from base class, remove 
    def __setitem__(self, key, value):
        assert hasattr(self, key), "Missing %s, options: %s" % (key, dir(self))
        setattr(self, key, value)
           
    def set_defaults(self):
        for parameter, _, default, _ in self.params:
            name = "%s_%s" % (self.name, parameter)
            setattr(self, name, default)                


def initialize_base_model(helper_function=AutoModelForSequenceClassification,
                          model_name="distilbert-base-uncased"):
    """
    Initialize a BERT model and corresponding tokenizer.

    Args:
        helper_function: The huggingface function that returns a BERT model.
        model_name: The name of the BERT model to use.
    """

    model = helper_function.from_pretrained(model_name, num_labels=2)

    # Freeze the model parameters

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        """
        Initialize a LoRA with two weight matrices whose product is the same as the original parameter matrix.
        """
        super().__init__()

        self.A = None
        self.B = None
        self.alpha = 0

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Complete the initialization of the two weight matrices
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the linear layer's original result, then add the low-rank delta
        """
        assert x.shape[-1] == self.in_dim, "Input dimension %s does not match input dimension %i" % (str(x.shape), self.in_dim)

        if len(x.shape) == 1:
            delta = torch.zeros(self.out_dim)
            output_dimension = torch.Size([self.out_dim])
        else:
            delta = torch.zeros((x.shape[0], self.in_dim))
            output_dimension = torch.Size((x.shape[0], self.out_dim))

        # Compute the low-rank delta

        return delta


class LinearLoRA(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, rank: int, alpha: float):
        """
        Initialize a Linear layer with LoRA adaptation.
        """
        super().__init__()
        self.linear = linear

        # Initialize the LoRA layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adatpation.
        """
        result = self.linear(x)

        # Add the LoRA delta
        return result

# TODO(jbg): Get rid of the hardcoded modules so that it generalizes to other models
def add_lora(model: torch.nn.Module, rank: int, alpha: float, 
             modules_to_adapt: Dict[str, List[str]] = {"attention": ["q_lin", "k_lin", "v_lin", "out_lin"], "ffn": ["lin1", "lin2"]}):
    """
    Add LoRA layers to a PyTorch model.  

    Args:
        model: The PyTorch model to adapt.
        rank: The rank of the LoRA matrices.
        alpha: The scaling factor for the LoRA matrices.
        modules_to_adapt: The key of the dictionary is the model component to adapt (e.g., "attention" or "ffn"), and the values are specific linear layers in that component to adapt.  Anything in this dictionary will be adapted, but anything else will remain frozen.
    """
    

    return model
                

class LoRABertBuzzer(Buzzer):
    def __init__(self, filename, run_length, num_guesses):
        super().__init__(filename, run_length, num_guesses)

        self._pipeline = None

    def initialize_model(self, model_name, rank, alpha):
        """
        Initialize the model and add LoRA layers.
        """

        self.model, self.tokenizer = initialize_base_model(model_name=model_name)
        add_lora(self.model.distilbert.transformer, rank, alpha)

    def add_data(self, questions):
        """
        We don't have features, so these become noops
        """
        
        None

    def build_features(self, history_length=0, history_depth=0):
        """
        We don't have features, so these become noops
        """
        
        None
        
    def dataset_from_questions(self, questions, answer_field="page"):
        """
        Build the dataframe so we can do fine tuning, requires us to get all the runs
        """

        # Todo: can we refactor this so this overrides build_features?  It's
        # basically doing the same thing.  Would be cleaner OOP.
        
        from eval import rough_compare
        from datasets import Dataset
        import pandas as pd
        
        metadata, answers, runs = self._clean_questions(questions, self.run_length, answer_field)

        all_guesses = {}
        logging.info("Building guesses from %s" % str(self._guessers.keys()))
        for guesser in self._guessers:
            all_guesses[guesser] = self._guessers[guesser].batch_guess(runs, self.num_guesses)
            logging.info("%10i guesses from %s" % (len(all_guesses[guesser]), guesser))
            assert len(all_guesses[guesser]) == len(runs), "Guesser %s wrong size" % guesser
            
        assert len(runs) == len(answers), "Runs (%i) don't match answers (%i)" % (len(questions), len(answers))
        assert len(metadata) == len(runs),  "Metadata (%i) don't match answers (%i)" % (len(metadata), len(answers))

        for guesser in all_guesses:
            assert len(all_guesses[guesser]) == len(runs), "Guesser %s length (%i) didn't match runs (%i)" % (guesser, len(all_guesses[guesser]), len(runs))
        
        dataset = []
        
        for guesser_id in all_guesses:
          for metadatum, answer, guesses, run in zip(metadata, answers, all_guesses[guesser_id], runs):
            example = {}

            for guess in guesses:
                if rough_compare(guess['guess'], answer):
                    correct = 1
                else:
                    correct = 0

                if answer is None:
                    answer = ""

                # You may want to add additional fields here
                example["text"] = "%0.2f [SEP] %s [SEP] %s" % (guess['confidence'], guess['guess'], run)
                example["label"] = correct
                example["answer"] = answer

                example["id"] = metadatum["qanta_id"]

                # We need the length for eval
                example["length"] = len(run)
                example["guess"] = guess['guess']
                example["confidence"] = guess['confidence']

                dataset.append(example)

        dataframe = pd.DataFrame(data=dataset)
        print(dataframe.head())
        return Dataset.from_pandas(dataframe)

    def predict(self, questions):
        if self._pipeline is None:
            from transformers import pipeline

            # TODO: set this up to use GPU based on device
            self._classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

            # TODO: Use explainability to generate some features
            self._classifier.coef_ = [[]]

        inputs = self.dataset_from_questions(questions)

        # Labels are LABEL_0 and LABEL_1
        predictions = [int(x["label"][-1]) for x in self._classifier(inputs["text"])]
        for question_index in range(len(questions)):
            questions[question_index]["run"] = inputs[question_index]["text"]
            questions[question_index]["guess"] = inputs[question_index]["guess"]
            questions[question_index]["answer"] = inputs[question_index]["answer"]
            questions[question_index]["id"] = inputs[question_index]["id"]
            
        return predictions, [], [], [pred==label for pred, label in zip(predictions, inputs["label"])], questions

    def train(self, finetune_dataset, eval_dataset):
        import numpy as np
        from transformers import DataCollatorWithPadding
        from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
        import evaluate
        
        from tqdm import tqdm

        accuracy = evaluate.load("accuracy")

        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True)
        
        dataset = {}
        dataset["train"] = finetune_dataset.map(preprocess_function, batched=True)
        dataset["eval"] = eval_dataset.map(preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)
                    
        training_args = TrainingArguments(  
            output_dir="models/lora_bert_buzzer",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()

    def save(self):
        torch.save(self.model, "%s.model" % self.filename)

    def load(self):
        self.model = torch.load("%s.model" % self.filename)


if __name__ == "__main__":
    import gzip

    import argparse
    import json

    from parameters import add_buzzer_params, add_question_params, load_guesser, load_buzzer, load_questions, add_general_params, setup_logging, add_guesser_params
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train_fold', type=str, default="../data/qanta.buzztrain.json.gz")
    # parser.add_argument('--test_fold', type=str, default="../data/qanta.buzzdev.json.gz")
    # parser.add_argument('--train_cache', type=str, default="../models/buzztrain_gpr_cache")
    # parser.add_argument('--test_cache', type=str, default="../models/buzzdev_gpr_cache")

    # parser.add_argument('--run_length', type=int, default=100)  
    # parser.add_argument('--limit', type=int, default=-1)

    # parser.add_argument('--rank', type=int, default=16)
    # parser.add_argument('--alpha', type=float, default=1.0)
    # parser.add_argument('--model_name', type=str, default="distilbert-base-uncased")

    guesser_params = add_guesser_params(parser)
    buzzer_params = add_buzzer_params(parser)
    question_params = add_question_params(parser)

    add_general_params(parser)
    flags = parser.parse_args()    

    flags = parser.parse_args()
    
    guesser = load_guesser(flags, guesser_params, load=True)    
    buzzer = load_buzzer(flags, buzzer_params)
    finetune_questions = load_questions(flags, secondary=False)
    dev_questions = load_questions(flags, secondary=True)
    
    setup_logging(flags)    

    # Train the model
    if finetune_questions:
      if flags.limit > 0:
        finetune_questions = finetune_questions[:flags.limit]
      finetune_dataset = buzzer.dataset_from_questions(finetune_questions)

    if dev_questions:
      if flags.limit > 0:
        dev_questions = dev_questions[:flags.limit]
      dev_dataset = buzzer.dataset_from_questions(dev_questions)
        
    buzzer.train(finetune_dataset, dev_dataset)
    buzzer.save()

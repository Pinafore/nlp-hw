import argparse
import pickle
import json
import time

from typing import List, Dict, Iterable, Optional, Tuple, NamedTuple
from collections import Counter
from random import random

import logging

from guesser import Guesser, kTOY_JSON

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

kUNK = '<unk>'

kPAD = '<pad>'

kTOY_VOCAB = [kUNK, "England", "Russia", "capital", "currency"]
kTOY_ANSWER = ["London", "Moscow", "Pound", "Rouble"]

class QuestionData(Dataset):
    def __init__(self, min_token_df, max_token_df, min_answer_freq):
        self.min_token_df = min_token_df
        self.max_token_df = max_token_df
        self.min_answer_freq = min_answer_freq

        self.answer_to_int = {}
        self.int_to_answer = []
        self.num_answers = 0

        self.vocab_to_int = {}
        self.int_to_vocab = []
        self.num_vocab = 0

        self.tokenizer = None

    def toy(self):
        """
        Initialize everything to toy parameters used in unit tests/demo.
        """
        

        self.int_to_answer = kTOY_ANSWER
        self.int_to_vocab = kTOY_VOCAB
        
        for ii, ww in enumerate(self.int_to_vocab):
            self.vocab_to_int[ww] = ii

        for ii, ww in enumerate(self.int_to_answer):
            self.answer_to_int[ww] = ii

        self.num_vocab = len(self.vocab_to_int)
        self.num_answers = len(self.answer_to_int)

        self.questions = [x['text'] for x in kTOY_JSON]
        self.answers = [x['page'] for x in kTOY_JSON]

        self.tokenizer = word_tokenize
        
    def save(self, path):
        """
        Save the question data to a file; we'll need this for DAN tokenization
        """
        for filename, data in [("%s.vocab.pkl" % path, "int_to_vocab"),
                               ("%s.answers.pkl" % path, "int_to_answer")]:
            with open(filename, 'wb') as f:
                logging.info("Writing DanGuesser to %s" % filename)
                pickle.dump(getattr(self, data), f)

    def load(self, path):
        """
        Load the question data (the mapping from words and answers to intigers) from a file; we'll need this for DAN tokenization
        """
        for filename, data in [("%s.vocab.pkl" % path, "int_to_vocab"),
                               ("%s.answers.pkl" % path, "int_to_answer")]:
            with open(filename, 'rb') as f:
                logging.info("Reading DanGuesser %s from %s" % (data, filename))
                setattr(self, data, pickle.load(f))

        for ii, ww in enumerate(self.int_to_vocab):
            self.vocab_to_int[ww] = ii

        for ii, ww in enumerate(self.int_to_answer):
            self.answer_to_int[ww] = ii

        self.num_answers = len(self.answer_to_int)
        self.num_vocab = len(self.vocab_to_int)

        assert self.num_vocab > 0
        assert self.num_answers > 0

    @staticmethod
    def vectorize(ex : Iterable[str], word2ind : dict[str, int]):
        """
        vectorize a single example based on the word2ind dict.
        Keyword arguments:
        exs: list of input questions-type pairs
        ex: tokenized question sentence (list)
        label: type of question sentence
        Output:  vectorized sentence(python list) and label(int)
        e.g. ['text', 'test', 'is', 'fun'] -> [0, 2, 3, 4]
        """

        vec_text = [0] * len(ex)

        #### modify the code to vectorize the question text
        #### You should consider the out of vocab(OOV) cases
        #### question_text is already tokenized
        ####Your code here


        return vec_text

    ###You don't need to change this funtion
    def __len__(self):
        return len(self.questions)

    ###You don't need to change this funtion
    def __getitem__(self, index : int):
        assert self.tokenizer is not None, "Tokenizer not set up"
        question = self.vectorize(self.tokenizer(self.questions[index]),
                                                 self.vocab_to_int)
        answer = self.answers[index]
        item = (question, answer)
        return item

    def initialize(self, answers, questions, answer_count, unk_drop, tokenizer=None):
        """
        Initialize the dataloader with the data.

        Keyword arguments:
        answers -- the answers to the questions
        questions -- the questions
        answer_count -- how many times each of the answers appear
        unk_drop -- remove this percentage of unknown answers
        tokenizer -- a function to tokenize text
        """
        self.tokenizer = tokenizer
        answer_lookup, vocab = self.create_indices(answers, questions, answer_count)
        self.int_to_vocab = vocab
        self.int_to_answer = answer_lookup

        for index, word in enumerate(vocab):
            self.vocab_to_int[word] = index

        for index, answer in enumerate(answer_lookup):
            self.answer_to_int[answer] = index

        self.num_vocab = len(self.int_to_vocab)
        self.num_answers = len(self.int_to_answer)

        assert len(self.int_to_answer) > 0, "Answer lookup not initialized"
        if kUNK in self.answer_to_int:
            assert self.int_to_answer[0] == kUNK, \
                "Assumed label 0 was %s, but got %s (total length %i)" % \
                (kUNK, self.int_to_answer[0], len(self.int_to_answer))

        self.answers = []
        self.questions = []

        unks_dropped = 0
        unks_kept = 0
        total_unks = 0
        for aa, qq in zip(answers, questions):
            answer_index = self.answer_to_int.get(aa, 0)
            # This second part of if statement is needed if there are
            # no UNKs in the datset.  Otherwise, we'll throw out
            # whatever random answer is 0
            if answer_index == 0 and self.int_to_answer[0] == kUNK:
                total_unks += 1
                if unk_drop > 0:
                    assert unk_drop <= 1.0
                    if random() < unk_drop:
                        logging.debug("Dropping answer %s" % aa)
                        unks_dropped += 1
                        continue
                    else:
                        unks_kept += 1
            self.answers.append(answer_index)
            self.questions.append(qq)

        total = len(self.answers)
        if unk_drop > 0:
            assert unks_kept + unks_dropped == total_unks
            if unks_dropped > 0:
                logging.info("Out of %i originals, %0.2f unks dropped and %0.2f kept (out of %i)" %
                             (len(answers), unks_dropped / (unks_kept + unks_dropped),
                              unks_kept / (unks_kept + unks_dropped), total_unks))
            else:
                logging.info("No unks dropped out of %i questions and %i unks" % (len(answer), total_unks))
                
        logging.info("%0.5f percent of %i questions are UNK, uniform would be %0.5f" % 
                         (sum(1 for x in self.answers if x==0) / total, total, 1 / len(set(self.answers))))

        self.answers = [self.answer_to_int.get(x, 0) for x in answers]
        self.questions = questions

    def copy_lookups(self, questions, dataset_with_lookups, answer_field):
        """
        Copy the lookup from another dataset and create lookups
        """
        question_text = Guesser.split_examples(questions, answer_field)

        assert dataset_with_lookups.num_answers > 1
        assert dataset_with_lookups.num_vocab > 1

        for attribute in ["int_to_vocab", "int_to_answer", "vocab_to_int", "answer_to_int", "num_vocab", "num_answers", "tokenizer"]:
            setattr(self, attribute, getattr(dataset_with_lookups, attribute))

        answer_to_questions = Guesser.split_examples(questions, answer_field)

        self.questions, answers = Guesser.filter_answers(answer_to_questions,
                                                         answer_lookup=dataset_with_lookups.answer_to_int)

        self.answers = [self.answer_to_int.get(x, 0) for x in answers]

    def create_indices(self, answers, questions, answer_count):
        """
        Create the answer map and vocabulary
        """

        # Sort the classes by decreasing frequency 
        valid_classes = [x for x, count in sorted(answer_count.items(),
                                                  key=lambda k: (-k[1], k[0]))
                         if count >= self.min_answer_freq and x is not None]

        if any(x for x in answer_count if not x in valid_classes):
            logging.debug("Found an unknown answer, so adding the unknown answer class")
            valid_classes = [kUNK] + valid_classes
        logging.info("Found %i answers that appear at least %i: %s" % (len(valid_classes),
                                                                       self.min_answer_freq,
                                                                       str(valid_classes)[:80]))

        # Build the phrase table
        word_count = Counter()
        tokenized_questions = [self.tokenizer(x) for x in questions]

        # if the df is expressed as proportion, convert to counts
        min_df = self.min_token_df
        if min_df < 1:
            min_df = int(len(tokenized_questions) * min_df)

        max_df = self.max_token_df
        if max_df <= 1:
            max_df = int(len(tokenized_questions) * max_df)

        for question in tokenized_questions:
            word_count.update(Counter(set(question)))
            
        # Sort by count first, then by alpha.  This is helpful for testing
        vocab = [x for x, count in sorted(word_count.items(), key=lambda k: (-k[1], k[0])) 
                 if count >= min_df and count <= max_df]
        vocab = [kUNK] + vocab

        logging.info("Looking for %i words with document frequency between %i and %i (total docs=%i) from %s " %
                         (len(vocab), min_df, max_df, len(tokenized_questions), str(word_count)[:250]))
        logging.info("Vocab examples: %s" % str(vocab)[:250])

        assert len(vocab) > 1, "Did not add any words to vocab"

        return valid_classes, vocab


class DanGuesser(Guesser):
    def __init__(self, filename, answer_field, min_token_df, max_token_df, min_answer_freq,
                 embedding_dimension, hidden_units, nn_dropout, grad_clipping, unk_drop,
                 batch_size, num_epochs, num_workers,
                 device):

        self.model_filename = filename
        self.train_data = QuestionData(min_token_df, max_token_df, min_answer_freq)
        self.eval_data = QuestionData(min_token_df, max_token_df, min_answer_freq)

        self.answer_field = answer_field
        self.embedding_dimension = embedding_dimension
        self.hidden_units = hidden_units
        self.nn_dropout = nn_dropout

        self.embedding_dimension = embedding_dimension
        self.hidden_units = hidden_units
        self.nn_dropout = nn_dropout
        self.grad_clipping = grad_clipping
        self.unk_drop = unk_drop

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.device = device

    def set_data_model(self, data, model):
        """
        Instead of training, set data and model directly.  Useful for unit tests.
        """

        self.dan_model = model
        self.train_data = data

    def vectorize(self, question: str) -> Iterable[int]:
        """
        This fuction in the DanGuesser takes a question and represents it in a form the DAN can ingest.

        We need to initialize the tokenizer for this to work.
        """
        assert self.phrase_tokenize is not None, "Need to initialize the tokenizer"
        return self.train_data.vectorize(self.phrase_tokenize(question), self.train_data.vocab_to_int)
    
    def train_dan(self):
        train_sampler = torch.utils.data.sampler.RandomSampler(self.train_data)
        dev_sampler = torch.utils.data.sampler.RandomSampler(self.eval_data)
        dev_loader = torch.utils.data.DataLoader(self.eval_data, batch_size=self.batch_size,
                                                 sampler=dev_sampler, num_workers=self.num_workers,
                                                 collate_fn=DanGuesser.batchify)
        self.best_accuracy = 0.0

        for epoch in range(self.num_epochs):
            train_loader = torch.utils.data.DataLoader(self.train_data,
                                                       batch_size=self.batch_size,
                                                       sampler=train_sampler,
                                                       num_workers=self.num_workers,
                                                       collate_fn=DanGuesser.batchify)
            self.run_epoch(train_loader, dev_loader)


    def train(self, training_data, answer_field, split_by_sentence,
              min_length=-1, max_length=-1, remove_missing_pages=True):
        answers_to_questions = Guesser.train(self, training_data, answer_field,
                                             split_by_sentence=split_by_sentence,
                                             min_length=min_length,
                                             max_length=max_length,
                                             remove_missing_pages=remove_missing_pages)

        answer_count = dict((x, len(y)) for x, y in
                            answers_to_questions.items())

        self.find_phrases(training_data)
        self.train_data.initialize(answers=self.answers,
                                   questions=self.questions,
                                   answer_count=answer_count,
                                   unk_drop=self.unk_drop,
                                   tokenizer=self.phrase_tokenize)

        num_classes = self.train_data.num_answers
        vocab_size = self.train_data.num_vocab
        self.dan_model = DanModel(model_filename=self.model_filename,
                                  n_classes=num_classes, device=self.device,
                                  vocab_size=vocab_size,
                                  emb_dim=self.embedding_dimension,
                                  n_hidden_units=self.hidden_units,
                                  nn_dropout=self.nn_dropout)

    def set_eval_data(self, dev_data):
        self.eval_data.copy_lookups(dev_data, self.train_data,
                                    self.answer_field)

    def __call__(self, question, n_guesses=1):
        model = self.dan_model

        question_text = torch.LongTensor([self.vectorize(question)])
        logging.debug("Vectorizing question %s to become %s" % (question, str(question_text)))
        question_length = torch.LongTensor([[len(question_text)]])

        logits = model.forward(question_text, question_length)
        logging.debug("Guess logits (query=%s len=%i): %s" % (question, len(logits), str(logits)))

        # If we don't have enough possible answers, constrain the number of
        # top results accordingly
        n_guesses = min(n_guesses, self.train_data.num_answers)
        assert n_guesses > 0, "Need to at least return 1 guess"
        
        values, indices = logits.topk(n_guesses)
        indices = [int(x) for x in torch.flatten(indices)]
        values = [float(x) for x in torch.flatten(values)]

        assert len(torch.flatten(logits)) == len(self.train_data.int_to_answer), \
            "Mismatched logit dimension %i vs %i" % \
            (len(torch.flatten(logits)), len(self.train_data.int_to_answer))

        guesses = []
        for guess_index in range(n_guesses):
            guess = self.train_data.int_to_answer[indices[guess_index]]

            if guess == kUNK:
                guess = ""

            guesses.append({"guess": guess, "confidence": values[guess_index]})
        return guesses

    def save(self):
        """
        Save the DanGuesser to a file
        """
        
        Guesser.save(self)

        torch.save(self.dan_model, "%s.torch.pkl" % self.model_filename)
        self.train_data.save(self.model_filename)

    def load(self):
        Guesser.load(self)
        
        self.dan_model = torch.load("%s.torch.pkl" % self.model_filename)
        self.train_data.load(self.model_filename)

    def run_epoch(self, train_data_loader, dev_data_loader,
                  checkpoint_every=50):
        """
        Train the current model
        Keyword arguments:
        train_data_loader: pytorch build-in data loader output for training examples
        dev_data_loader: pytorch build-in data loader output for dev examples
        """

        model = self.dan_model
        model.train()
        optimizer = torch.optim.Adamax(self.dan_model.parameters())
        criterion = nn.CrossEntropyLoss()
        print_loss_total = 0
        epoch_loss_total = 0
        start = time.time()

        #### modify the following code to complete the training funtion
        for idx, batch in enumerate(train_data_loader):
            question_text = batch['text'].to(self.device)
            question_len = batch['len']
            labels = batch['labels']

            #### Your code here
            loss = None
            # You'll need to actually compute a loss and update the optimizer

            assert loss is not None, "The loss has not been defined"
            
            clip_grad_norm_(model.parameters(), self.grad_clipping)
            print_loss_total += loss.data.numpy()
            epoch_loss_total += loss.data.numpy()

            if idx % checkpoint_every == 0:
                print_loss_avg = print_loss_total / checkpoint_every

                logging.info('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time()- start))
                print_loss_total = 0
                curr_accuracy = evaluate(dev_data_loader, self.dan_model, self.device)

                if accuracy < curr_accuracy:
                    torch.save(model, "%s.torch.pkl" % self.model_filename)
                    accuracy = curr_accuracy

        logging.info('Final accuracy=%f' % accuracy)
        return accuracy


    ###You don't need to change this funtion
    @staticmethod
    def batchify(batch):
        """
        Gather a batch of individual examples into one batch,
        which includes the question text, question length and labels
        Keyword arguments:
        batch: list of outputs from vectorize function
        """

        question_len = list()
        label_list = list()
        for ex in batch:
            question_len.append(len(ex[0]))
            label_list.append(ex[1])

        target_labels = torch.LongTensor(label_list)
        x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
        for i in range(len(question_len)):
            question_text = batch[i][0]
            vec = torch.LongTensor(question_text)
            x1[i, :len(question_text)].copy_(vec)
        q_batch = {'text': x1, 'len': torch.LongTensor(question_len), 'labels': target_labels}
        return q_batch

def evaluate(data_loader, model, device):
    """
    evaluate the current model, get the accuracy for dev/test set
    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """

    model.eval()
    num_examples = 0
    error = 0
    for idx, batch in enumerate(data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        # Call the model to get the logits
        # You'll need to update the code here        

        top_n, top_i = logits.topk(1)
        num_examples += question_text.size(0)
        error += torch.nonzero(top_i.squeeze() - torch.LongTensor(labels)).size(0)
    accuracy = 1 - error / num_examples
    return accuracy






class DanModel(nn.Module):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    #### You don't need to change the parameters for the model


    def __init__(self, model_filename, n_classes, device, vocab_size, emb_dim=50,
                 activation=nn.ReLU(), n_hidden_units=50, nn_dropout=.5):
        super(DanModel, self).__init__()
        self.model_filename = model_filename
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout
        logging.info("Creating embedding layer for %i vocab size, %i classes with %i dimensions (hidden dimension=%i)" % \
                         (self.vocab_size, n_classes, self.emb_dim, n_hidden_units))
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        self.linear1 = nn.Linear(emb_dim, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_classes)

        # Create the actual prediction framework for the DAN classifier.

        # You'll need combine the two linear layers together, probably
        # with the Sequential function.  The first linear layer takes
        # word embeddings into the representation space, and the
        # second linear layer makes the final prediction.  Other
        # layers / functions to consider are Dropout, ReLU.
        # For test cases, the network we consider is - linear1 -> ReLU() -> Dropout(0.5) -> linear2

        # To allow the code to pass the unit tests, we created a WRONG network.
        # You will need to replace it with something that actually works correctly.

        self.network = nn.Sequential(
            nn.Linear(emb_dim, n_classes)
        )
        

        # To make this work on CUDA, you need to move it to the appropriate
        # device
        self.network.to(device)

    def average(self, text_embeddings, text_len):
        """
        Given a batch of text embeddings and a tensor of the corresponding lengths, compute the average of them.

        text_embeddings: embeddings of the words
        text_len: the corresponding number of words
        """

        average = torch.zeros(text_embeddings.size()[0], text_embeddings.size()[-1])

        # You'll want to finish this function.  You don't have to use it in
        # your forward function, but it's a good way to make sure the
        # dimensions match and to use the unit test to check your work.

        return average

    def forward(self, input_text, text_len, is_prob=False):
        """
        Model forward pass, returns the logits of the predictions.

        Keyword arguments:
        input_text : vectorized question text, were each word is represented as an integer
        text_len : batch * 1, text length for each question
        is_prob: if True, output the softmax of last layer
        """

        logits = torch.FloatTensor([0.0] * self.n_classes)

        # Complete the forward funtion.  First look up the word embeddings.
        # Then average them
        # Before feeding them through the network

        if is_prob:
            logits = self._softmax(logits)

        return logits



# You basically do not need to modify the below code
# But you may need to add funtions to support error analysis

if __name__ == "__main__":
    from params import load_questions, add_guesser_params, add_general_params, add_question_params, setup_logging

    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Question Type')
    add_general_params(parser)
    add_guesser_params(parser)
    add_question_params(parser)

    flags = parser.parse_args()
    setup_logging(flags)

    #### check if using gpu is available
    cuda = not flags.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    logging.info("Device=", device)

    ### Load data

    train_exs = load_questions(flags)
    dev_exs = load_questions(flags, secondary=True)

    # dev_exs = load_data(args.dev_file, -1)
    # test_exs = load_data(args.test_file, -1)

    logging.info("Loaded %i train examples" % len(train_exs))
    logging.info("Loaded %i dev examples" % len(dev_exs))
    logging.info("Example: %s" % str(train_exs[0]))

    ### Create vocab
    dg = DanGuesser(filename=flags.DanGuesser_filename, answer_field=flags.guesser_answer_field, min_token_df=flags.DanGuesser_min_df, max_token_df=flags.DanGuesser_max_df,
                    min_answer_freq=flags.DanGuesser_min_answer_freq, embedding_dimension=flags.DanGuesser_embedding_dim,
                    hidden_units=flags.DanGuesser_hidden_units, nn_dropout=flags.DanGuesser_dropout,
                    grad_clipping=flags.DanGuesser_grad_clipping, unk_drop=flags.DanGuesser_unk_drop,
                    batch_size=flags.DanGuesser_batch_size, num_epochs=flags.DanGuesser_num_epochs,
                    num_workers=flags.DanGuesser_num_workers, device=device)
    dg.train(train_exs, flags.guesser_answer_field, flags.tfidf_split_sentence, flags.tfidf_min_length,
                 flags.tfidf_max_length)
    dg.set_eval_data(dev_exs)


    dg.train_dan()
    print(dg("Name this Italian author of The Path to the Nest of Spiders, Invisible Cities, and If on a winter's night a traveler"))
    dg.save()

    dg = DanGuesser(filename=flags.DanGuesser_filename, answer_field=flags.guesser_answer_field, min_token_df=flags.DanGuesser_min_df, max_token_df=flags.DanGuesser_max_df,
                    min_answer_freq=flags.DanGuesser_min_answer_freq, embedding_dimension=flags.DanGuesser_embedding_dim,
                    hidden_units=flags.DanGuesser_hidden_units, nn_dropout=flags.DanGuesser_dropout,
                    grad_clipping=flags.DanGuesser_grad_clipping, unk_drop=flags.DanGuesser_unk_drop,
                    batch_size=flags.DanGuesser_batch_size,
                    num_epochs=flags.DanGuesser_num_epochs, num_workers=flags.DanGuesser_num_workers,
                    device=device)
    dg.load()
    print(dg("Name this Italian author of The Path to the Nest of Spiders, Invisible Cities, and If on a winter's night a traveler"))

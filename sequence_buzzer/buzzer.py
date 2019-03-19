import argparse
import nltk
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os
from os import path
from typing import List, Dict, Iterable, Optional, Tuple, NamedTuple
from collections import defaultdict
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import random

from dataset_util import QuizBowlDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_



###You don't need to change anything in this class
class TfidfGuesser:
    '''
    This is the guesser class; here we use Tfidf. 
    
    Methods: 
    
    train: To train the Tfidf guesser.
    guess: Use the trained model to make guesses on a text of question. 
    save: Save the trained model to location specified in GUESSER_MODEL_PATH.
    load: Can load a previously saved trained model from the location specified in GUESSER_MODEL_PATH.
    '''
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None

    def train(self, training_data) -> None:
        '''
        Must be passed the training data - list of questions from the QuizBowlDataset class
        '''
        questions, answers = [], []
        for ques in training_data:
            questions.append(ques.sentences)
            answers.append(ques.page)

        answer_docs = defaultdict(str)
        for q, ans in zip(questions, answers):
            text = ' '.join(q)
            answer_docs[ans] += ' ' + text

        x_array = []
        y_array = []
        for ans, doc in answer_docs.items():
            x_array.append(doc)
            y_array.append(ans)

        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        
        # below hypermaters for the vectorizer may be tuned.
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=.9).fit(x_array)
        
        self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        '''
        Will output the top n guesses specified by max_n_guesses acting on a list of question texts (strings).
        '''
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]
        
        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])

        return guesses

    def save(self, guesser_model_path):
        with open(guesser_model_path, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    @classmethod
    def load(cls, guesser_model_path):
        with open(guesser_model_path, 'rb') as f:
            params = pickle.load(f)
            guesser = TfidfGuesser()
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.i_to_ans = params['i_to_ans']
            return guesser


###You don't need to change this funtion
def get_trained_guesser_model(questions):
    '''
    questions is the QuizbowlDataset object's output, returned from its data() method (check out dataset_util.py)
    '''
    print('Training the Guesser...')
    tfidf_guesser = TfidfGuesser()
    tfidf_guesser.train(questions)
    print('---Guesser is Trained and Ready to be Used---')
    return tfidf_guesser


###You don't need to change this funtion
def generate_ques_data_for_guesses(questions, char_skip = 50):
    '''
    First, we generate the data in the form we need for the guesser to then act on. 
    Input:
        questions: list of Question class type objects (dataset_util) - containing the question text, answer, etc.
        char_skip: characters to skip before producing next guesser output (each question gets divided into partial
        snippets, for e.g - first 50 characters, first 100 characters... for guessing to occur, and buzz output
        to be predicted. See also dataset_util for an example of the question.runs method.)
    Output:
        ques_nums: Each element is a question number
        char_indices: Each element is list of character indices upto which current question text has been recorded
        question_texts: Each element is a list of snippets of the current question, increasing in length
        answers: Each element is the answer page to the current question
        question_lens: Each element is the length of each question in terms of number of snippets the question got
                        divided into via use of char_skip
    '''

    ques_nums = []
    char_indices = []
    question_texts = []
    answers = []
    question_lens = []

    print("Preparing Data for Guessing; # of questions: " + str(len(questions)))

    for q in questions:
        qnums_temp, answers_temp, char_inds_temp, curr_ques_texts = [], [], [], []
        for text_run, char_ix in zip(*(q.runs(char_skip))):
            curr_ques_texts.append(text_run)
            qnums_temp.append(q.qanta_id)
            answers_temp.append(q.page)
            char_inds_temp.append(char_ix)
        ques_nums.append(qnums_temp)
        char_indices.append(char_inds_temp)
        question_texts.append(curr_ques_texts)
        answers.append(answers_temp)
        question_lens.append(len(curr_ques_texts))
        
    return ques_nums, answers, char_indices, question_texts, question_lens


###You don't need to change this funtion
def generate_guesses_and_scores(model, questions, max_guesses, char_skip = 50):
    '''
    This function generates the guesses and corresponding score for the text snippets of a question.
    For example, consider the (rough) input -
    [[['Name this co'], ['Name this country where Alan'], ['Name this country where Alan Turing was born.']],
    [['For ten points'], ['For ten points name the chess'], ['For ten points name the chess engine that chall']]]
    There are two questions above, each question consists a number of text snippets
    
    The main output of this function - guesses_and_scores - will look something like this assuming max_guesses=3 -
    [[[('Little_Brown_Foxes', 0.1435), ('Jerry_Seinfeld', 0.1332), ('India', 0.1198)], 
      [('United_States', 0.1335), ('England', 0.1212), ('Canada', 0.1011)],
      [('England', 0.1634), ('United_States', 0.1031), ('France', 0.0821)]],
      
      [[('Little_Brown_Foxes', 0.1435), ('Jerry_Seinfeld', 0.1332), ('India', 0.1198)], 
      [('Chess', 0.1335), ('Gary_Kasparov', 0.1212), ('Anton_Karpov', 0.1011)],
      [('Deep_Blue', 0.1634), ('Gary_Kasparov', 0.1031), ('Chess', 0.0821)]]]
      
    Inputs:
    model: The guesser model
    questions: list of Question class type objects (dataset_util) - containing the question text, answer, etc.
    max_guesses: number of guesses to generate
    char_skip: characters to skip before producing next guesser output (each question gets divided into partial
        snippets, for e.g - first 50 characters, first 100 characters... for guessing to occur, and buzz output
        to be predicted. See also dataset_util for an example of the question.runs method.)
    '''
    
    
    #get the neccesary data
    qnums, answers, char_indices, ques_texts, ques_lens = generate_ques_data_for_guesses(questions, char_skip)
    print('Guessing...')
    
    '''
    Consider roughly 10-12 snippets per question. The tfidf guesser is very fast; it would be a waste to generate
    guesses for just one question at a time. However, trying to get guesses for the whole dataset in one go (
    after flattening out the list) would mean at least 50,000 text snippets for just 5000 questions - memory issues.
    So, we go for something in between below, taking all snippets of 250 questions at a time resulting in 
    roughly 3000 snippets for which the guesser acts every time. This required slicing list of questions to take
    250 questions every time, flattening this list to get the list of all snippets, getting the guesses, and finally,
    'de-flatten' the guesses and scores list using the list of original question lengths to maintain the format we
    want.
    '''
    
    guesses_and_scores = []
    for i in tqdm(range(0, len(ques_texts), 250), mininterval=5, maxinterval=60):
        try:
            q_texts_temp = ques_texts[i:i+250]
            q_lens = ques_lens[i:i+250]
        except:
            q_texts_temp = ques_texts[i:]
            q_lens = ques_lens[i:]
            
        #flatten
        q_texts_flattened = []
        for q in q_texts_temp:
            q_texts_flattened.extend(q)
            
        #store guesses for the flattened questions
        print('Guessing directly on %d text snippets together' %len(q_texts_flattened))
        flattened_guesses_scores = model.guess(q_texts_flattened, max_guesses)
        
        #de-flatten using question lengths, and add guesses and scores 
        #(now corresponding to one question at a time) to the main list
        j = 0
        for k in q_lens:
            guesses_and_scores.append(flattened_guesses_scores[j:j+k])
            j = j + k
        
    assert len(guesses_and_scores)==len(ques_texts)
    
    print('Done Generating Guesses and Scores.')
    
    return qnums, answers, char_indices, ques_texts, ques_lens, guesses_and_scores
    

#You need to write code inside this function
def create_feature_vecs_and_labels(guesses_and_scores, answers, n_guesses):
    '''
    This function takes in the guesses and scores output from the function above and uses it to 
    create feature vector corresponding to every question (will be a vector of vectors since every question
    contains multiple snippets), and also creates the buzz label for each snippet.
    
    For this homework, we go for a very simple feature vector. We simply take the probabilties or scores
    of the top n_guesses and use it as our feature vector for the text snippet. Label for each snippet is 
    1 if the top guess is same as the answer, 0 otherwise. Implement this below
    
    Inputs:
        guesses_and_scores: List of top guesses and corresponding scores by guesser model; output of (and 
                            described in) the generate_guesses_and_scores function.
        answers: Each element is the actual answer to the current question.
        n_guesses: Number of top guesses to consider.
        
    HINT/EXAMPLE:
    Consider example output (guesses_and_scores) described in generate_guesses_and_scores function.
    Corresponding 'xs' will be -
    [[[0.1435, 0.1332, 0.1198], 
      [0.1335, 0.1212, 0.1011],
      [0.1634, 0.1031, 0.0821]],
      
      [[0.1435, 0.1332, 0.1198], 
       [0.1335, 0.1212, 0.1011],
       [0.1634, 0.1031, 0.0821]]]
    Corresponding 'ys' will be - 
    [[0, 0, 1], [0, 0, 1]]
    (if n_guesses=3 and actual answers to the two questions were 'England' and 'Deep_Blue' resp.)
    '''
    xs, ys = [], []
    for i in range(len(answers)):
        guesses_scores = guesses_and_scores[i]
        ans = answers[i]
        length = len(ans)
        labels = []
        prob_vec = []
        
        for j in range(length):
            ## YOUR CODE BELOW
            


        
        xs.append(np.array(prob_vec))
        ys.append(np.array(labels))
    exs = list(zip(xs, ys))
    return exs


###You don't need to change this funtion
def batchify(batch):
    """
    Gather a batch of individual examples into one batch, 
    which includes the question feature vec, question length, and labels 
    (feature vec and labels generated in create_feature_vecs_and_labels function)
    """

    question_len = list()
    label_list = list()
    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])
    
    '''
    Padding the labels - unequal length sequences for sequenial data like we have. 
    Since actual labels are 0/1 - we pad with -1, and will use this when 'masking' labels during loss and
    accuracy evaluation.
    '''
    target_labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(y) for y in label_list], padding_value=-1).t()
    
    #dimension is dimension of every feature vector = n_guesses in this homework setting
    dim = batch[0][0].shape[1]
    
    #similar padding happens for the feature vectors, with vector of all zeros appended.
    x1 = torch.FloatTensor(len(question_len), max(question_len), dim).zero_()
    for i in range(len(question_len)):
        question_feature_vec = batch[i][0]
        vec = torch.FloatTensor(question_feature_vec)
        x1[i, :len(question_feature_vec)].copy_(vec)
    q_batch = {'feature_vec': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
    return q_batch


class QuestionDataset(Dataset):
    """
    Pytorch data class for questions
    """

    ###You don't need to change this funtion
    def __init__(self, examples):
        self.questions = []
        self.labels = []

        for qq, ll in examples:
            self.questions.append(qq)
            self.labels.append(ll)
    
    ###You don't need to change this funtion
    def __getitem__(self, index):
        return self.questions[index], self.labels[index]
    
    ###You don't need to change this funtion
    def __len__(self):
        return len(self.questions)


###You don't need to change this funtion
def accuracy_fn(logits, labels):
    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.contiguous()
    labels = labels.view(-1)
    
    # flatten all predictions
    logits = logits.contiguous()
    logits = logits.view(-1, 2)   #2 is the number of labels
    
    #create mask - remember, we padded using -1 for our labels in batchify
    mask = (labels > -1).float()
    
    #these are the actual number of examples ignoring padded stuff
    num_examples = int(torch.sum(mask).data)
    
    #get the non-zero indices of the mask - these are the corresponding indices in logits/labels
    #that contain data of value (rest is just padded)
    indices = torch.nonzero(mask.data).squeeze()
    
    #get the logits corresponding to non-padded values as given by non-zero indices of mask
    logits = torch.index_select(logits, 0, indices)
    
    #get the logits corresponding to non-padded values as given by non-zero indices of mask
    labels = torch.index_select(labels, 0, indices)
    
    top_n, top_i = logits.topk(1)
    error = torch.nonzero(top_i.squeeze() - torch.LongTensor(labels)).size(0)
    
    return error, num_examples


###You don't need to change this funtion
def loss_fn(outputs, labels):
    
    #to compute cross entropy loss
    outputs = F.log_softmax(outputs, dim=1)
    
    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.contiguous()
    labels = labels.view(-1)  
    
    # flatten all predictions
    outputs = outputs.contiguous()
    outputs = outputs.view(-1, 2)   #2 is the number of labels
    
    # mask out 'PAD' tokens
    mask = (labels > -1).float()

    # the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).data)

    # pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels]*mask

    # cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs)/num_tokens


#You need to write code inside this function
def train(args, model, train_data_loader, dev_data_loader, device):
    """
    Train the current model
    Keyword arguments: 
    args: arguments, here we use checkpoint value
    model: model to be trained
    train_data_loader: pytorch build-in data loader output for training examples
    dev_data_loader: pytorch build-in data loader output for dev examples
    device: cpu or gpu
    """

    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()

    #### modify the following code to complete the training funtion
    
    best_train_acc, best_dev_acc = 0.0, 0.0

    for idx, batch in enumerate(train_data_loader):
        question_feature_vec = batch['feature_vec'].to(device)
        question_len = batch['len'].to(device)
        labels = batch['labels'].to(device)

        #### Your code here ----
        
        #zero out
        
        
        #get output from model
        
        
        #use loss_fn defined above to calculate loss
        
        
        #use accuracy_fn defined above to calculate 'error' and number of examples ('num_examples') used to
        #calculate accuracy below.
        
        
        #backprop
        
        

        ###Your code ends ---
        accuracy = 1 - error/num_examples
        clip_grad_norm_(model.parameters(), 5) 
        print_loss_total += loss.data.numpy()
        epoch_loss_total += loss.data.numpy()
        
        if (idx+1) % args.checkpoint == 0 and idx > 0:
            print_loss_avg = print_loss_total / args.checkpoint
            
            dev_acc = evaluate(dev_data_loader, model, device)

            print('number of steps: %d, train loss: %.5f, train acc: %.3f, dev acc: %.3f, time: %.5f' 
                  % (idx+1, print_loss_avg, accuracy, dev_acc, time.time()- start))
            print_loss_total = 0
            if accuracy>best_train_acc:
                best_train_acc = accuracy
            if dev_acc>best_dev_acc:
                best_dev_acc = dev_acc

    return best_train_acc, best_dev_acc


#You need to write code inside this function
def evaluate(data_loader, model, device):
    """
    evaluate the current model, get the accuracy for dev/test set
    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu or gpu
    """

    model.eval()
    total_num_examples = 0
    total_error = 0
    for idx, batch in enumerate(data_loader):
        question_feature_vec = batch['feature_vec'].to(device)
        question_len = batch['len'].to(device)
        labels = batch['labels'].to(device)

        ####Your code here ---
        
        #get the output from the model

        
        #get error, num_examples using accuracy_fn defined previously

        
        #update total_error and total_num_examples




    accuracy = 1 - total_error/total_num_examples
    return accuracy


#You need to write code inside functions of this class
class RNNBuzzer(nn.Module):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    We use a LSTM for our buzzer.
    """

    #### You don't need to change the parameters for the model

    #n_input represents dimensionality of each specific feature vector, n_output is number of labels
    def __init__(self, n_input=10, n_hidden=50, n_output=2, dropout=0.5):
        super(RNNBuzzer, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.n_output = n_output
        
        ### Your Code Here --- 
        #define lstm layer, going from input to hidden. Remember to have batch_first=True.
        
        
        #define linear layer going from hidden to output.
        
        
        
        #---you can add other things like dropout between two layers, but do so in forward function below,
        #as we have to perform an extra step on the output of LSTM before going to linear layer(s).
    def forward(self, X, X_lens):
        """
        Model forward pass, returns the logits of the predictions.
        
        Keyword arguments:
        input_text : vectorized question text 
        text_len : batch * 1, text length for each question
        is_prob: if True, output the softmax of last layer
        """
        
        #get the batch size and sequence length (max length of the batch)
        #dim of X: batch_size x batch_max_len x input feature vec dim
        batch_size, seq_len, _ = X.size()
        
        ###Your code here --
        #Get the output of LSTM - (output dim: batch_size x batch_max_len x lstm_hidden_dim)
        
        
        #reshape (before passing to linear layer) so that each row contains one token 
        #essentially, flatten the output of LSTM 
        #dim will become batch_size*batch_max_len x lstm_hidden_dim
        

        
        #Get logits from the final linear layer
        
        
        
        #--shape of logits -> (batch_size, seq_len, self.n_output)
        return logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options for RNN Buzzer')
    parser.add_argument('--guesser_model_path', type=str, default='tfidf.pickle', help='path for saving the trained guesser model')
    parser.add_argument('--n_guesses', type=int, default=10, help='number of top guesses to consider for creating feature vector')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size for the training of the buzzer')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of Epochs for training the buzzer')
    parser.add_argument('--guesser_train_limit', type=int, default=-1, help = 'Limit the data used to train the Guesser (total is around 90,000)')
    parser.add_argument('--buzzer_train_limit', type=int, default=5000, help = 'Limit the data used to train the Buzzer (total is around 16,000)')
    parser.add_argument('--buzzer_dev_limit', type=int, default=500, help = 'Limit the dev data used to evaluate the buzzer (total is 1161)')
    parser.add_argument('--buzzer_test_limit', type=int, default=500, help = 'Limit the test data used to evaluate the buzzer (total is 1953)')
    parser.add_argument('--guesser_saved_flag', type=bool, default=False, help='flag indicating use of saved guesser model or training one')
    parser.add_argument('--buzz_data_saved_flag', type=bool, default=False, help='flag indicating using saved data files to train buzzer, or generating them')
    parser.add_argument('--see_test_accuracy', type=bool, default=False, help='flag indicating seeing test result FINALLY after tuning on dev')
    parser.add_argument("--char_skip", type=int, default=50, help='number of characters to skip after which buzzer should predict')
    parser.add_argument('--checkpoint', type=int, default=50)

    args = parser.parse_args()



    if args.guesser_train_limit<0:
        train_guess_questions = QuizBowlDataset(guesser=True, split='train').data()
    else:
        train_guess_questions = QuizBowlDataset(guesser=True, split='train').data()[:args.guesser_train_limit]
        
    if args.buzzer_train_limit<0:
        train_buzz_questions = QuizBowlDataset(buzzer=True, split='train').data()
    else:
        train_buzz_questions = QuizBowlDataset(buzzer=True, split='train').data()[:args.buzzer_train_limit]
        
    if args.buzzer_dev_limit<0:
        dev_buzz_questions = QuizBowlDataset(buzzer=True, split='dev').data()
    else:
        dev_buzz_questions = QuizBowlDataset(buzzer=True, split='dev').data()[:args.buzzer_dev_limit]

    if args.buzzer_test_limit<0:
        test_buzz_questions = QuizBowlDataset(buzzer=True, split='test').data()
    else:
        test_buzz_questions = QuizBowlDataset(buzzer=True, split='test').data()[:args.buzzer_test_limit]


    if args.guesser_saved_flag:
        guesser_model = TfidfGuesser().load(args.guesser_model_path)
    else:
        guesser_model = get_trained_guesser_model(train_guess_questions)
        guesser_model.save(args.guesser_model_path)
        print('Guesser Model Saved! Use --guesser_saved_flag=True when you next run the code to load the trained guesser directly.')



    if args.buzz_data_saved_flag:
        train_exs = np.load('train_exs.npy')
        dev_exs = np.load('dev_exs.npy')
        test_exs = np.load('test_exs.npy')
    else:
        print('Generating Guesses for Training Buzzer Data')
        train_qnums, train_answers, train_char_indices, train_ques_texts, train_ques_lens, train_guesses_and_scores =  \
                                                                generate_guesses_and_scores(guesser_model, train_buzz_questions, args.n_guesses, char_skip=args.char_skip)
        print('Generating Guesses for Dev Buzzer Data')
        dev_qnums, dev_answers, dev_char_indices, dev_ques_texts, dev_ques_lens, dev_guesses_and_scores =    \
                                                                generate_guesses_and_scores(guesser_model, dev_buzz_questions, args.n_guesses, char_skip=args.char_skip)
        print('Generating Guesses for Test Buzzer Data')
        test_qnums, test_answers, test_char_indices, test_ques_texts, test_ques_lens, test_guesses_and_scores =   \
                                                                generate_guesses_and_scores(guesser_model, test_buzz_questions, args.n_guesses, char_skip=args.char_skip)
        train_exs = create_feature_vecs_and_labels(train_guesses_and_scores, train_answers, args.n_guesses)
        #print(len(train_exs))
        dev_exs = create_feature_vecs_and_labels(dev_guesses_and_scores, dev_answers, args.n_guesses)
        #print(len(dev_exs))
        test_exs = create_feature_vecs_and_labels(test_guesses_and_scores, test_answers, args.n_guesses)
        #print(len(test_exs))
        np.save('train_exs.npy', train_exs)
        np.save('dev_exs.npy', dev_exs)
        np.save('test_exs.npy', test_exs)
        print('The Examples for Train, Dev, Test have been SAVED! Use --buzz_data_saved_flag=True next time when you \
                run the code to use saved data and not generate guesses again.')


    train_dataset = QuestionDataset(train_exs)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=0,
                                           collate_fn=batchify)

    dev_dataset = QuestionDataset(dev_exs)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, sampler=dev_sampler, num_workers=0,
                                           collate_fn=batchify)

    test_dataset = QuestionDataset(test_exs)
    test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=0,
                                           collate_fn=batchify)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = RNNBuzzer()
    model.to(device)
    print(model)


    for epoch in range(args.num_epochs):
        print('start epoch %d' %(epoch+1))
        train_acc, dev_acc = train(args, model, train_loader, dev_loader, device)


    if args.see_test_accuracy:
        print('The Final Test Set Accuracy')
        print(evaluate(test_loader, model, device))

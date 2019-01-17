import argparse
import os
import json

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

from sklearn.feature_extraction.text import TfidfVectorizer

class QuizBowlData(Dataset):
    """
    Quiz Bowl Dataset
    """

    def __init__(self, json_file, root_dir, vectorizer,
                       is_train=True, transform=None):
        self.root_dir = root_dir
        self.vectorizer = vectorizer
        self.transform = transform
        self.tfidf = None
        self.is_train = is_train
        
        if json_file is None:
            self.json = None
        else:
            with open(os.path.join(self.root_dir, json_file)) as infile:
                self.json = json.load(infile)["questions"]

    def num_classes(self):
        """
        The number of classes in the dataset
        """
        assert self.tfidf is not None, "tf-idf must be initialized! (run vectorize)"
        return len(self.label_set)

    def example_size(self):
        assert self.tfidf is not None, "tf-idf must be initialized! (run vectorize)"        
        return len(self.vectorizer.vocabulary_)
            
    def vectorize(self, json=None):
        """
        Turn the json documents into a matrix appropriate for
        features used by Pytorch.  Must set the labels and tfidf data
        members of this class as a side-effect.

        Keyword arguments:
        json -- an optional json string to bypass the file
        """

        # Complete this function to prepare your data
        
        if json is None: 
            json = self.json
        else:
            self.json = json
        

        # Convert to COO format to create a torch matrix
        

    def __len__(self):
        assert self.tfidf is not None, "tf-idf must be initialized! (run vectorize)"        
        return len(self.json)
        
    def __getitem__(self, idx):
        assert self.tfidf is not None, "tf-idf must be initialized! (run vectorize)"
        
        sample = self.tfidf[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample, self.labels[idx]

def setup(train, learn_rate):
    model = nn.Linear(train.example_size(),
                      train.num_classes())

    # Loss and optimizer
    # nn.CrossEntropyLoss() computes softmax internally
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    return model, optimizer
    
def step(iteration, model, criterion, optimizer, questions, labels):
    """
    Take a step of a the model, compute the gradient, and backpropagate that through the parameters of the model.
    """

    # Complete this function
    
    # Compute the output and loss of the model on these examples
        
    # Create the loss and update the parameters

    # Print the progress
    if iteration % 100 == 1:
        print ('Step {}, Loss: {:.4f}' 
                   .format(i+1, loss.item()))
    return loss
    
def accuracy(model, loader):
    """
    Given predictions and labels, compute the accuracy of the model
    """
    with torch.no_grad():
        correct = 0
        total = 0

    return correct / total

def top_words(model, labels, vectorizer):
    """
    Take a model and output the highest weight words for each class.  
    """

    vocab = vectorizer.vocabulary_
    weights = {}

    return weights
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--root_dir", help="QB Dataset for training",
                           type=str, default='../',
                           required=False)    
    argparser.add_argument("--train_dataset", help="QB Dataset for training",
                           type=str, default='qanta.train.json',
                           required=False)
    argparser.add_argument("--test_dataset", help="QB Dataset for test",
                           type=str, default='qanta.dev.json',
                           required=False)
    argparser.add_argument("--epochs", help="Number of passes over dataset",
                           type=int, default=3, required=False)
    argparser.add_argument("--vocab_size", help="Number of features",
                           type=int, default=5000, required=False)
    argparser.add_argument("--learn_rate", help="Learning rate",
                            type=float, default=0.001, required=False)
    argparser.add_argument("--max_df", help="Max document frequency",
                           type=float, default=0.5, required=False)
    argparser.add_argument("--min_df", help="Documents features must appear in",
                           type=int, default=5, required=False)
    argparser.add_argument("--batch_size", help="Batch size",
                           type=int, default=100, required=False)          
    argparser.add_argument("--limit", help="Number of training documents",
                           type=int, default=-1, required=False)    
    args = argparser.parse_args()

    vectorizer = TfidfVectorizer(max_df=args.max_df,
                                 max_features=args.vocab_size,
                                 min_df=args.min_df, stop_words='english',
                                 use_idf=True)
    train_dataset = QuizBowlData(args.train_dataset, args.root_dir, vectorizer)
    test_dataset = QuizBowlData(args.test_dataset, args.root_dir, vectorizer,
                                is_train=False)

    train_dataset.vectorize()
    vocab_size = len(vectorizer.vocabulary_)
    print("Vocab size: %i" % vocab_size)
    test_dataset.vectorize()

    # Data loader (input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, 
                                              shuffle=False)

    # Logistic regression model
    criterion = nn.CrossEntropyLoss()  
    model, optimizer = setup(train_dataset, args.learn_rate)
    print("Dimension size: " + str(model.weight.size()))

    # Train the model
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (questions, labels) in enumerate(train_loader):
            step(i, model, criterion, optimizer, questions, labels)
            
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    acc = accuracy(model, test_loader)
    print('Accuracy of the model on test questions: %f' % acc)

    # Write out a quick summary of what the model learned
    weights = top_words(model, train_dataset.label_set, vectorizer)
    for ii in weights:
        top_words = sorted(weights[ii].items(), key=lambda x: -x[1])[:10]
        print("%s\t: " % ii + " ".join("%s:%0.2f" % (x, float(y)) for x, y in top_words))
    
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

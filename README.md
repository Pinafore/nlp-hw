# nlp-hw
Homework for NLP course at University of Maryland

Shared Data
==================
The homeworks use the dataset of questions.  You can use the Makefile to download the files:
```cd nlp-hw
make qanta.train.json
```
The above data can also be used for the project since the files contain the quizbowl dataset questions.

You can also download the supporting paragraphs or sentences corresponding to every sentence of a quizbowl question. This is particularly helpful for developing reading comprehension style systems on quizbowl. 

For the page, paragraph index, sentence index, and correct answer span information:
```cd nlp-hw
make qanta.train.evidence.json
```

For the page, paragraph index, sentence index, and sentence text information:
```cd nlp-hw
make qanta.train.evidence.text.json
```

Prerequisites
==================
I assume that you have pytorch, sklearn, and nltk installed.  It's easy to do this with [Anaconda](https://anaconda.org/pytorch/pytorch).

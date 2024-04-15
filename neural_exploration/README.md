

The goal of this homework is for you to use Pytorch to create a model
and run it on the data we've been using (qanta.buzztrain,
qanta.guesstrain, etc.) to do something interesting.

This is an open-ended assignment, so you can be as creative as you'd
like.  While you're welcome to try to do a better job of answering
questions (guessing) or calibrating models (buzzing), you're also
welcome to:

* Try to predict the category of a question

* Try to predict a year the question was written

* Use the data for language modeling

* Run a topic model on the data

Or anything else that has a well-defined loss.

Since we're at the end of the year, you should be thinking about your
projects, so I'd encourage you to pick a project that will relate to
your course project.

So if you're writing questions, you could fine-tune a retrieval system
to find passages given a question to help you do a better job of writing questions.

If you're building a system with four other people, you could find
four subproblems that you need to solve and each of you could focus on
creating a system for that.  E.g., if you are working toward an
overall QA system you could have one person work on retrieval, one on
reranking, one on extraction, and one on buzzing.

It's also okay to work in groups on this project, but the effort
required will scale with the number in the group.

Concrete Ideas
==============

This ammount of freedom might be intimidating, so if you need an
explicit template of what to do, here are some ideas:

* Use a character-level n-gram model to generate QB questions:

https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
(It's going to be horrible, but it will be fun)

* Classify the category of a question:

https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

* Learn a RNN / LSTM buzzer:

https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

The DAN Homework
================

Originally, the plan for this homework was to implement the DAN model,
and we provided source code here to help you with it:
https://github.com/Pinafore/nlp-hw/tree/master/dan

You're welcome to continue working on it or to use some of the code,
but I think we now realize that the introduction of the new loss
function plus a nearest-neighbor lookup was too much.  So we don't
recommend doing this model for your homework.

Grading
=======

This will not be graded very rigorously.  

FAQ
====

**Q: Can I use other people's code?  How about other people's models?**

You can use other's code if you cite it.  But you must train/finetune
a model on the course data yourself.  In other words, you can't
download code and data and do nothing else.  A model must be trained.






What to Do
=============

You'll need to complete three functions:
* QuizBowlData.vectorize
* step
* accuracy

I'll talk about each of these and what you have to do for them.

QuizBowlData.vectorize
--------------
This function will create data in the form that PyTorch can use.  This
is a matrix with a row per document and a column per feature.  The
vectorizer will do much of the work for you, but it will likely be a
little work to figure out how it works and what it gives you as an
output.

The vectorizer will figure out what your vocabulary should be using
the *fit_transform* function.  But once your vocabulary is fixed, for
test data, you don't want your vocabulary to grow.  You'll then want
to run the *transform* function (look at the *is_train* argument of
the *vectorize* function to know which to use).  Run that on a list of
documents, and that will produce a sparse matrix.  Unfortunately, it's
in the [wrong
format](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csc_matrix.tocoo.html).
You'll need to turn that [sparse
matrix](https://pytorch.org/docs/stable/sparse.html) into a dense
matrix for input into Pytorch.  This will get stored into
*self.tfidf*.

Next, you'll need to create the label for each document.  Figure out
the list of labels, and then assign an integer for the label of each
document.  This will get stored into *self.labels*.

Why are we spending so much time on data wrangling, when you're
chomping at the bit to do deep learning?  Unfortunately, much of NLP
is getting your data in the right form so that you can actually use
it.  This will force you to know what the data look like so that you
can then do the classification correctly.

step
----------

For the step function, you'll need to compute the forward pass, take
the loss, and then have the optimizer take a step to update the model
parameters.  This should be a relatively short function and should
look a lot like the example covered in the lecture.  Not trying to
trick you here!

accuracy
----------

For computing the accuracy, you'll need to make a prediction on every
document, figure out which class has the highest probability, and then
see whether that prediction matches the correct label.

Write Up
=================

For each category, figure out which words are the most useful features
for that class.  Are there changes to the vectorizer that would
improve the classification.
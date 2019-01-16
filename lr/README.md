

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
the *fit_transform* function.  Run that on a list of documents, and
that will produce a sparse matrix.  You'll need to turn that sparse
matrix into a dense matrix for input into Pytorch.

Next, you'll need to create the label for each document.  Figure out
the list of labels, and then assign an integer for the label of each
document.

Why are we spending so much time on data wrangling, when you're
chomping at the bit to do deep learning?  Unfortunately, much of NLP
is getting your data in the right form so that you can actually use
it.

step
----------

For the step function, you'll need to 
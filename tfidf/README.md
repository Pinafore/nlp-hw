
This first real programming homework will be relatively easy.  The most important parts
of this homework are to understand how I do homeworks for the course,
my coding style, and the level of coding required for the course.

But we still want to do something relevant to the material of the
course, so we'll be computing [tf-idf
representations](https://onlinelibrary.wiley.com/doi/full/10.1002/bult.274)
of documents.

Big Picture
=============

The goal here is to create tf-idf representations of documents.  You
won't do it in this homework "for real", but this allows you to take a
query and find the closest document.

But to define "closest", we need to define a vector space over
documents.  So what are the dimensions of this vector space?  Words!

So the very first step of this process is going to map an individual
word to an integer.  For example, "enterprise" is 1701 and "valjean"
is 24601.  For this homework, we're going to call this process the
"vocabulary lookup": look up the integer that will represent it, and
return that.  That integer will then corespond to the dimension of a
very large vector space.

Take a look at the function ``vocab_lookup`` to see what that looks
like.  You don't need to implement that exact function, but you will
need to figure out what goes into the vocab.  Simple, right?

Now, of course there are some complications.  
  
First complication: what if after we've seen a new word that wasn't in the
vocabulary?  Anything that isn't mapped to the vocabulary will then
become the "unknown token".
 
That leads to a second complication: we need to compute statistics for
   how often documents have unknown words.  If we add every single
   word in our training set to the vocabulary, then there won't be any
   unknown words and thus no statistics about unknown words.

So what do unknown words look like?  Think about Zipf's law.  There
are very few frequent words but many infrequent words.  So we're
likely to have most of the frequent words in our vocabulary.  That
means we're missing infrequent words.  

So what words that we have seen
will look most like the unknown words that we'll see in the future?
Our least frequent words!  So we'll use the ``unk_cutoff`` argument to
turn all of the words that we initially saw into the unknown token
``kUNK = "<UNK>"``.

Okay, so that's our vocabulary.  We also need to compute statistics
for tf-idf.  We can't do everything at once, so we'll need to do two
passes over the data.  The first pass will count how many times we see
each word in the training set (using the function ``train_seen``), and
the second pass will compute term and document frequencies (using the
function ``add_document``).  In between those two passes, we'll
finalize our vocabulary to decide the integer lookup of all of our
words (the ``finalize`` function).

Then, you should have everything you need to compute---for a new
document or query---the tf-idf representation in the ``doc_tfidf``
function!



What to Do
=============

# Code (20 Points)

You'll need to complete several functions in the TfIdf class:
* constructor
* train_seen
* finalize
* add_document
* term_freq
* inv_docfreq

I'll talk about each of these and what you have to do for them.  Each
of these should be fairly easy to do.  If you find yourself spending
hours on one of these, you're probably overthinking it or doing
something that Python can do for you.

They're listed roughly in the order that you should complete them, but
you obviously need to think about all of them first before you can
start.

constructor
--------------

You don't need to do too much here except for creating datastructures
that you may need to count things up later.  I'd suggest taking a look
at NLTK's
[FreqDist](http://www.nltk.org/api/nltk.html?highlight=freqdist) and/or
refresh your memory on Python
[collections](https://docs.python.org/3/library/collections.html).

train_seen
----------

Here, you'll need to take the *string* representations that you've
seen and keep track of how often you've seen them.

finalize
----------

Once we've done a scan over all of the documents, we can create a
vocabulary, taking all of the words that have appeared more than or
equal to unk_cutoff times into the vocabulary.  You make want to free
up the memory you used for train_seen here, but not necessary for
getting a good grade.

The most important thing is that after this function has run, the
vocab data member of TfIdf should map words (in string form) to their
vocabulary id (an integer).

add_document
---------

Take in a document and keep track of the words that you see in
appropriate datastructures so that the next two functions work.

term_freq
----------

Return the frequency of a word.

inv_docfreq
-------------

Return the inverse document frequency of a word.


Write Up (5 Points)
=================

What are words that, specifically for this collection, appear in a lot
of documents and are thus not helpful query terms?

# Submission Instructions


1. Submissions will be made on Gradescope.

2. You will submit a zip file containing your code (tfidf.py) and the PDF for
the write-up. If you check your code against your own test cases, you can add
the file (which will be like test.py) containing your own test cases in the
zip too.

The code will run against the public test cases (the ones you can already see
in the given test.py file) on the server and you can see those results. You
should make sure you pass these cases before the submission deadline.

# Hints

1.  Remember to first make sure you pass all of the local unit tests.
2.  NLTK's FreqDist and Python's built in Counter are your friends.
3.  Make sure you use the right log.
4.  Look at the main function to see how the code will be called and
    then figure out what's missing from the code.

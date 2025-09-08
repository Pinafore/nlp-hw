
This first real programming homework but the implementation will be
relatively easy.  The most important parts of this homework (and
probably the hardest, to be honest) are to understand how I do
homeworks for the course, my coding style, and the level of coding
required for the course.

But we still want to do something relevant to the material of the
course, so we'll be computing [tf-idf
representations](https://onlinelibrary.wiley.com/doi/full/10.1002/bult.274)
of documents.

Big Picture
=============
The goal here is to create tf-idf representations of documents.  You
won't do it in this homework "for real" (i.e., in a way that can scale
up to thousands of documents), but this allows you to take a query and
find the closest document ... the math and code will be legit, just
not very efficient.

But to define "closest", we need to define a vector space over
documents.  So what are the dimensions of this vector space?  Words!
First whitespace-separated tokens, then from a BPE tokenizer.

In other words, we'll be building a matrix `_doc_vectors` in the code.
It will have rows corresponding to the number of documents and columns
corresponding to the number of tokens present in the document (with an
index for each type).  But at first we don't know how many words or
documents we have, so we'll do this in three steps:
 * Find out what words we're working with
 * Find out how many documents we have
 * Build the matrix

First pass: whitespace tokenization
---------------

The code we've provided initially uses whitespace tokenization, you
should get this working first before switching to BPE tokenization.

The good think about whitespace tokenization is that it doesn't need
to be trained.

If you use the code as is, it will tokenize documents just fine:

    list(guesser.tokenize("University of Maryland"))
    [327, 276, 256]

The problem is that it doesn't know all the words (like, say,
Maryland).

    guesser._vocab._id_to_word[256]
    '<UNK>'

We'll fix that in a second, but let's first implement tf-idf with
whitespace tokenization.

After we figure out the vocabulary, we'll call the `finalize_vocab` and
`finalize_docs` functions to tell the code to not let that change any
more.  After that point, we won't be able to tell the code that we've
seen new words or new documents.

So the very first step of this process is going to map an individual
word to an integer.  For example, "enterprise" is 1701 and "valjean"
is 24601.  For this homework, we're going to call this process the
"vocabulary lookup": look up the integer that will represent it, and
return that.  That integer will then corespond to the dimension of a
very large vector space.


What are the Documents?
---------------------

Okay, so that's our vocabulary (first attempt, at least).  We also need to compute statistics
for tf-idf.  We can't do everything at once, so we'll need to do two
passes over the data.  The first pass will count how many times we see
each word in the training set (using the function ``train_seen``), and
the second pass will compute term and document frequencies (using the
function ``scan_document``).  In between those two passes, we'll
finalize our vocabulary to decide the integer lookup of all of our
words (the ``finalize_vocab`` function).

Then, you should have everything you need to compute---for a new
document or query---the tf-idf representation in the ``doc_tfidf``
function!

Improving the Vocabulary
=============

We want to get rid of the unknown tokens, so rather than doing whitespace tokenization, we'll use BPE to discover what the words are for our TF-IDF approach.  But because TF-IDF is pretty simple, let's implement that first completely before jumping into BPE.

What to Do
=============

# Code

You'll need to complete several functions in the
`toytokenizer_guesser.py` file:
* `scan_document`
* `__call__`
* `inv_docfreq`
* `constructor`
* `frequent_bigram`
* `merge_tokens`
* `train`
* `tokenize`
* `add_from_merge` (optional)

I'll talk about each of these and how they fit together.  Just reading
this REAMDE will not tell you enough about how to implement them,
you'll also need to look at the doc strings.  Each
of these should be fairly easy to do.  If you find yourself spending
hours on one of these, you're probably overthinking it or doing
something that Python can do for you.

They're listed roughly in the order that you should complete them, but
you obviously need to think about all of them first before you can
start.

scan_document
---------

We want to compute idf, and the first stage is to store how often
words are seen in each document.  You'll need to do that here.

Take in a document and keep track of the words that you see in
appropriate datastructures so that the next two functions work.


constructor (for ToyTokenizerGuesser)
--------------

You don't need to do too much here except for creating datastructures
that you may need to count things up later.  I'd suggest taking a look
at NLTK's
[FreqDist](http://www.nltk.org/api/nltk.html?highlight=freqdist) and/or
refresh your memory on Python
[collections](https://docs.python.org/3/library/collections.html).

You'll probably need to add something here to handle what happens when
you merge BPEs and when you scan documents.

inv_docfreq
-------------

If you've stored how often words appear in every document during
`scan_doc` you can use that information now to compute invert document
frequencies.


train (first time)
-----
You will need update the train function to update your vocabulary and to keep track of how many times words appear in documents.

embed
======
After your system is trained, you can now embed documents by creating a tf-idf vector for a query document (in addition to all of the training documents).

`__call__`
-------------

Before you start coding this, remember what this function did in the last
homework: given a query, it needs to find the training item closest to the
query.  To do that, you need to do three things: turn the query into a vector,
compute the similarity of that vector with each row in the matrix, and return
the metadata associated with that row.

We've helped you out by structuring the code so that it should be easy for you
to complete it.  `question_tfidf` is the vector after you embed it.  This code
is already done for you (assuming you've completed `inv_docfreq` already).

Then you'll need to go through the rows in `self._doc_vectors` and find the
closest row.  Call whatever the closest is `best` and return the appropriate
metadata.  This is implemented for you already.


> ⚠️After you've done this, your code should work with the whitespace
> tokenizer and give you halfway decent results.  Play around with
> your code and make sure you're getting reasonable answers (even if
> you can't pass the unit tests yet).

frequent_bigram
----------

Now, start on BPE.  The first step is to find the most frequent byte
pairs in a token sequence.

merge_tokens
----------

Once you know what the frequent bigrams are, you need to turn those
into a new token.  There are unit tests for the examples we worked
through in class.

tokenize
-----

After you've *trained* the tokenizer, you'll need to run it on new
strings.  That's a separate function.

add_from_merge
----------

I added another function to the Vocab class to make things easier to
keep track of new tokens.  (This is optional).

train (second time)
-----------

I'm listing train again because you'll need to update it once you've done BPE tokenization.


Running Your Code
=================

First, make sure you pass the unit tests.  Inititally, it will look
like this:

         100%|███████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 45100.04it/s]
         100%|███████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 99864.38it/s]
         0it [00:00, ?it/s]
         0it [00:00, ?it/s]
         0it [00:00, ?it/s]
         100%|██████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 335544.32it/s]
         100%|██████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 508400.48it/s]
         Creating initial vocabulary: 100%|██████████████████████████████████████████████| 4/4 [00:00<00:00, 16794.01it/s]
         Creating document freq: 100%|███████████████████████████████████████████████████| 4/4 [00:00<00:00, 29641.72it/s]
         Creating document vecs: 100%|███████████████████████████████████████████████████| 4/4 [00:00<00:00, 16529.28it/s]
         100%|██████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 578524.69it/s]
         100%|██████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 671088.64it/s]
         Creating initial vocabulary: 100%|██████████████████████████████████████████████| 4/4 [00:00<00:00, 30559.59it/s]
         Creating document freq: 100%|███████████████████████████████████████████████████| 4/4 [00:00<00:00, 32201.95it/s]
         Creating document vecs: 100%|███████████████████████████████████████████████████| 4/4 [00:00<00:00, 27730.94it/s]
         100%|████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 592673.39it/s]
         100%|██████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 898779.43it/s]
         Creating initial vocabulary: 100%|████████████████████████████████████████████| 13/13 [00:00<00:00, 15150.31it/s]
         Creating document freq: 100%|█████████████████████████████████████████████████| 13/13 [00:00<00:00, 22852.45it/s]
         Creating document vecs: 100%|█████████████████████████████████████████████████| 13/13 [00:00<00:00, 18464.60it/s]
         100%|███████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 1090519.04it/s]
         100%|█████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 1143901.09it/s]
         Creating initial vocabulary: 100%|████████████████████████████████████████████| 13/13 [00:00<00:00, 18860.59it/s]
         Creating document freq: 100%|█████████████████████████████████████████████████| 13/13 [00:00<00:00, 24255.32it/s]
         Creating document vecs: 100%|█████████████████████████████████████████████████| 13/13 [00:00<00:00, 17784.07it/s]
         F
         ======================================================================
         FAIL: test_chinese_vocab_extra_credit (__main__.TestSequenceFunctions.test_chinese_vocab_extra_credit)
         This is for extra credit, you're not required to implement this.
         ----------------------------------------------------------------------
         Traceback (most recent call last):
           File "/Users/jbg/repositories/nlp-hw/tfidf/toytokenizer_test.py", line 50, in test_chinese_vocab_extra_credit
             self.assertNotEqual(vocab.lookup_index('马'), vocab.lookup_index('里'))
         AssertionError: 256 == 256
         
         ======================================================================
         FAIL: test_df (__main__.TestSequenceFunctions.test_df)
         ----------------------------------------------------------------------
         Traceback (most recent call last):
           File "/Users/jbg/repositories/nlp-hw/tfidf/toytokenizer_test.py", line 189, in test_df
             self.assertAlmostEqual(self.guesser.inv_docfreq(currency), 0.30102999566398114)
         AssertionError: 1.0 != 0.30102999566398114 within 7 places (0.6989700043360189 difference)
         
         ======================================================================
         FAIL: test_embed (__main__.TestSequenceFunctions.test_embed)
         ----------------------------------------------------------------------
         Traceback (most recent call last):
           File "/Users/jbg/repositories/nlp-hw/tfidf/toytokenizer_test.py", line 123, in test_embed
             self.assertAlmostEqual(test_doc[currency], 0.033447777295997905, delta=0.01)
         AssertionError: 0.5 != 0.033447777295997905 within 0.01 delta (0.4665522227040021 difference)
         
         ======================================================================
         FAIL: test_empty_df (__main__.TestSequenceFunctions.test_empty_df)
         ----------------------------------------------------------------------
         Traceback (most recent call last):
           File "/Users/jbg/repositories/nlp-hw/tfidf/toytokenizer_test.py", line 142, in test_empty_df
             self.assertAlmostEqual(self.guesser.inv_docfreq(word_a), 0.1249387366082999, delta=0.01)
         AssertionError: 1.0 != 0.1249387366082999 within 0.01 delta (0.8750612633917001 difference)
         
         ======================================================================
         FAIL: test_frequent (__main__.TestSequenceFunctions.test_frequent)
         ----------------------------------------------------------------------
         Traceback (most recent call last):
           File "/Users/jbg/repositories/nlp-hw/tfidf/toytokenizer_test.py", line 40, in test_frequent
             self.assertEqual(frequent[test], self.guesser.frequent_bigram(utf8_answers[test]), test)
         AssertionError: (44, 32) != None : elements
         
         ======================================================================
         FAIL: test_no_end_in_frequent (__main__.TestSequenceFunctions.test_no_end_in_frequent)
         ----------------------------------------------------------------------
         Traceback (most recent call last):
           File "/Users/jbg/repositories/nlp-hw/tfidf/toytokenizer_test.py", line 74, in test_no_end_in_frequent
             self.assertEqual(self.guesser.frequent_bigram(sample), (5, 5))
         AssertionError: None != (5, 5)
         
         ======================================================================
         FAIL: test_replace (__main__.TestSequenceFunctions.test_replace)
         ----------------------------------------------------------------------
         Traceback (most recent call last):
           File "/Users/jbg/repositories/nlp-hw/tfidf/toytokenizer_test.py", line 100, in test_replace
             self.assertNotEqual(candidate, None)
         AssertionError: None == None
         
         ======================================================================
         FAIL: test_tokenize (__main__.TestSequenceFunctions.test_tokenize)
         Test tokenization after training the tokenizer.
         ----------------------------------------------------------------------
         Traceback (most recent call last):
           File "/Users/jbg/repositories/nlp-hw/tfidf/toytokenizer_test.py", line 181, in test_tokenize
             self.assertEqual(reconstruction, reference[doc_id], "Tokens: %s" % str(tokens))
         AssertionError: 'This*capital*of*England*.' != 'This *capital *of *En*gl*an*d*.*<ENDOFTEXT>'
         - This*capital*of*England*.
         + This *capital *of *En*gl*an*d*.*<ENDOFTEXT>
          : Tokens: <generator object ToyTokenizerGuesser.whitespace_tokenize at 0x161fc5c40>
         
         ======================================================================
         FAIL: test_tokenize_wo_merge (__main__.TestSequenceFunctions.test_tokenize_wo_merge)
         If we don't train the tokenizer, the tokenization should just be characters.
         ----------------------------------------------------------------------
         Traceback (most recent call last):
           File "/Users/jbg/repositories/nlp-hw/tfidf/toytokenizer_test.py", line 163, in test_tokenize_wo_merge
             self.assertEqual(reconstruction, reference[doc_id], "Tokenization of %s bad results %s!=%s." % (doc["text"], reconstruction, reference[doc_id]))
         AssertionError: '<UNK>*<UNK>*<UNK>*<UNK>*.' != 'T*h*i*s* *c*a*p*i*t*a*l* *o*f* *E*n*g*l*a*n*d*.*<ENDOFTEXT>'
         - <UNK>*<UNK>*<UNK>*<UNK>*.
         + T*h*i*s* *c*a*p*i*t*a*l* *o*f* *E*n*g*l*a*n*d*.*<ENDOFTEXT>
          : Tokenization of This capital of England. bad results <UNK>*<UNK>*<UNK>*<UNK>*.!=T*h*i*s* *c*a*p*i*t*a*l* *o*f* *E*n*g*l*a*n*d*.*<ENDOFTEXT>.
         
         ======================================================================
         FAIL: test_train_vocab (__main__.TestSequenceFunctions.test_train_vocab)
         ----------------------------------------------------------------------
         Traceback (most recent call last):
           File "/Users/jbg/repositories/nlp-hw/tfidf/toytokenizer_test.py", line 61, in test_train_vocab
             self.assertEqual(self.guesser._vocab.examples(3),
         AssertionError: Lists differ: ['Massachusetts', '<ENDOFTEXT>', 'circulation'] != ['which states that bad money drives ', 'Fo[68 chars]at ']
         
         First differing element 0:
         'Massachusetts'
         'which states that bad money drives '
         
         - ['Massachusetts', '<ENDOFTEXT>', 'circulation']
         + ['which states that bad money drives ',
         +  'For 10 points, name this author of ',
         +  'New England state with capital at ']
         
         ----------------------------------------------------------------------
         Ran 11 tests in 0.014s
         
         FAILED (failures=10)
		 
After all of the required tests have been passed, it will look like this:

         100%|███████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 40329.85it/s]
         100%|██████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 102300.10it/s]
         0it [00:00, ?it/s]
         0it [00:00, ?it/s]
         0it [00:00, ?it/s]
         100%|██████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 305040.29it/s]
         100%|██████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 559240.53it/s]
         Creating initial vocabulary: 100%|██████████████████████████████████████████████| 4/4 [00:00<00:00, 16070.13it/s]
         Creating document freq: 100%|███████████████████████████████████████████████████| 4/4 [00:00<00:00, 30338.55it/s]
         Creating document vecs: 100%|███████████████████████████████████████████████████| 4/4 [00:00<00:00, 22580.37it/s]
         100%|██████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 559240.53it/s]
         100%|██████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 578524.69it/s]
         Creating initial vocabulary: 100%|██████████████████████████████████████████████| 4/4 [00:00<00:00, 31536.12it/s]
         Creating document freq: 100%|███████████████████████████████████████████████████| 4/4 [00:00<00:00, 35696.20it/s]
         Creating document vecs: 100%|███████████████████████████████████████████████████| 4/4 [00:00<00:00, 32201.95it/s]
         100%|████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 757304.89it/s]
         100%|██████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 993387.79it/s]
         Creating initial vocabulary: 100%|████████████████████████████████████████████| 13/13 [00:00<00:00, 19038.39it/s]
         Creating document freq: 100%|█████████████████████████████████████████████████| 13/13 [00:00<00:00, 14281.29it/s]
         Creating document vecs: 100%|█████████████████████████████████████████████████| 13/13 [00:00<00:00, 12680.45it/s]
         100%|████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 924168.68it/s]
         100%|██████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 471859.20it/s]
         Creating initial vocabulary: 100%|████████████████████████████████████████████| 13/13 [00:00<00:00, 15402.81it/s]
         Creating document freq: 100%|█████████████████████████████████████████████████| 13/13 [00:00<00:00, 14889.66it/s]
         Creating document vecs: 100%|█████████████████████████████████████████████████| 13/13 [00:00<00:00, 12960.77it/s]
         .
         ======================================================================
         FAIL: test_chinese_vocab_extra_credit (__main__.TestSequenceFunctions.test_chinese_vocab_extra_credit)
         This is for extra credit, you're not required to implement this.
         ----------------------------------------------------------------------
         Traceback (most recent call last):
           File "/Users/jbg/repositories/GPT3QA/toytokenizer_test.py", line 50, in test_chinese_vocab_extra_credit
             self.assertNotEqual(vocab.lookup_index('马'), vocab.lookup_index('里'))
             ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         AssertionError: 256 == 256
         
         ----------------------------------------------------------------------
         Ran 11 tests in 0.058s
         
         FAILED (failures=1)

There's one unit test for the Chinese extra credit.  You're not
required to pass that one, so it's okay if you leave that failed.

However, these tests aren't very realistic.  You will also want to run
the code with some small English data that we've provided.  You can do
this via the main method of the `toyttokenizer_guesser.py` file:

         100%|████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 254794.17it/s]
         100%|██████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 639809.08it/s]
         INFO:root:Trained with 13 questions and 13 answers filtered from 13 examples
         Creating initial vocabulary: 100%|████████████████████████████████████████████| 13/13 [00:00<00:00, 14379.21it/s]
         DEBUG:root:339 vocab elements, including: ['Massachusetts', '<ENDOFTEXT>', 'circulation', 'University', 'university', 'Prejudice', 'principle', 'Synagogue', 'identify', 'composer']
         Creating document freq: 100%|█████████████████████████████████████████████████| 13/13 [00:00<00:00, 23331.60it/s]
         Creating document vecs: 100%|█████████████████████████████████████████████████| 13/13 [00:00<00:00, 15966.60it/s]
         DEBUG:root:Document matrix is 13 by 339, has 207 non-zero entries
         ----------------------
         This capital of England. [{'question': 'For 10 points, name this New England state with capital at Augusta.', 'guess': 'Maine', 'confidence': 0.0}]
         ----------------------
         The author of Pride and Prejudice. [{'question': 'For 10 points, name this New England state with capital at Augusta.', 'guess': 'Maine', 'confidence': 0.0}]
         ----------------------
         The composer of the Magic Flute. [{'question': 'For 10 points, name this New England state with capital at Augusta.', 'guess': 'Maine', 'confidence': 0.0}]
         ----------------------
         The economic law that says 'good money drives out bad'. [{'question': 'For 10 points, name this New England state with capital at Augusta.', 'guess': 'Maine', 'confidence': 0.0}]
         ----------------------
         Located outside Boston, the oldest University in the United
		 States. [{'question': 'For 10 points, name this New England
		 state with capital at Augusta.', 'guess': 'Maine',
		 'confidence': 0.0}]
		 
Because of the template code we gave you, this will always return the
first document as the answer (which is bad)!  Once you've done the
assignment, you'll get more reasonable answers:

        100%|████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 429338.20it/s]
         100%|██████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 686340.65it/s]
         INFO:root:Trained with 13 questions and 13 answers filtered from 13 examples
         Creating initial vocabulary: 100%|████████████████████████████████████████████| 13/13 [00:00<00:00, 12322.25it/s]
         DEBUG:root:517 vocab elements, including: ['which states that bad money drives ', 'For 10 points, name this author of ', 'New England state with capital at ', ' states that bad money drives ', 'states that bad money drives ', 'For 10 points, name this au', 'For 10 points, name this ', 'states that bad money dr', 'Pride and Prejudice.', 'e with capital at ']
         Creating document freq: 100%|█████████████████████████████████████████████████| 13/13 [00:00<00:00, 14333.85it/s]
         Creating document vecs: 100%|█████████████████████████████████████████████████| 13/13 [00:00<00:00, 12395.08it/s]
         DEBUG:root:Document matrix is 13 by 517, has 270 non-zero entries
         ----------------------
         This capital of England. [{'question': 'For 10 points, name this city in New England, the capital of Massachusetts.', 'guess': 'Boston', 'confidence': np.float64(0.322173250335654)}]
         ----------------------
         The author of Pride and Prejudice. [{'question': 'For 10 points, name this author of Pride and Prejudice.', 'guess': 'Jane_Austen', 'confidence': np.float64(0.5731848450898768)}]
         ----------------------
         The composer of the Magic Flute. [{'question': 'Name this composer who wrote a famous requiem and The Magic Flute.', 'guess': 'Wolfgang_Amadeus_Mozart', 'confidence': np.float64(0.4823367232261605)}]
         ----------------------
         The economic law that says 'good money drives out bad'. [{'question': 'For 10 points, name this economic principle which states that bad money drives good money out of circulation.', 'guess': "Gresham's_law", 'confidence': np.float64(0.2274456308089948)}]
         ----------------------
         Located outside Boston, the oldest University in the United
		 States. [{'question': 'For 10 points, name this city in New
		 England, the capital of Massachusetts.', 'guess': 'Boston',
		 'confidence': np.float64(0.24435411459791467)}]
		 
You can get reasonable results with the whitespace tokenization
(before you do anything with BPE).  We strongly encourage you to do
that first (i.e., implement TODO).

What's "Good Enough"?
==================
This is not an open ended homework, so you really just need to:
complete the tf-idf caclulations and build the BPE dictionary.

The rest is extra credit.

Multibyte Character Support (1 points)
=====================

Support multibyte unicode characters during merges.  There's a Chinese
test case to see if you've implemented this.

Analysis Extra Credit (2 points)
=====================
Some questions that you might want to consider:
 * We had the whitespace tokenizer but then threw it away.  
     1. Is BPE actually better than whitespace tokenization 
     2. Is your tokenization better when you don't allow merges across whitespace?
   What do you gain?  What do you lose?
 * Can you improve the efficiency (e.g., in searching or in merging)
 * Our default `min_frequency` was 2, is that the right choice?


Efficiency Extra Credit (2 Points)
=================

Make your code go faster for both training and testing.  One way to do
this is to override the `batch_guess` function.

You can also get extra credit for submitting new test cases that test
functionality that the supplied tests miss (i.e., if you discover a
bug that isn't covered by our tests, you can get extra credit by
submitting a test that covers it).

# Submission Instructions


1. Submissions will be made on Gradescope.

2. You will submit toyttokenizer_guesser.py. If you check your code against your own test cases,
you can add the file (which will be like test.py) containing your own
test cases in the zip too.

The code will run against the public test cases (the ones you can already see
in the given test.py file) on the server and you can see those results. You
should make sure you pass these cases before the submission deadline.

For the analysis extra credit, upload a PDF of your analysis.

# Hints

1.  Remember to first make sure you pass all of the local unit tests.
2.  NLTK's FreqDist and Python's built in Counter are your friends.
3.  Make sure you use the right base log.
4.  Look at the main function to see how the code will be called and
    then figure out what's missing from the code.

FAQ
============

**Q: Can we add more methods to the Vocab class?**

**A:** Yes, so long as the existing methods don't change.  For
instance, you may want to add a helper function to keep track of
updated vocabulary from a merge.

**Q: Why is there a ``page`` field and an ``answer`` field.  Which one do I use?**

**A:** As you can see, the ``answer`` field has inconsistent formatting and is sometimes ambiguious.   To make things a little more sane, we map all of the answers to Wikipedia page titles.  This makes it so that rather than having to guess exactly the crazy formatting of the answer line, the Guesser just needs to match up to the correct underlying entity.  Not all questions have pages, which does cause a problem, but we're going to ignore that issue for a while, as most of them do have pages associated.

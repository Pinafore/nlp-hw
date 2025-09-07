
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

   >>> list(guesser.tokenize("University of Maryland"))
   [327, 276, 256]

The problem is that it doesn't know all the words (like, say,
Maryland).

   >>> guesser._vocab._id_to_word[256]
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

We want to get rid of the 

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
this via the main method of the `toytfidf_guesser.py` file:

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

Analysis Extra Credit
=====================
Some questions that you might want to consider:
 * We had the whitespace tokenizer but then threw it away.  
     1. Is BPE actually better than whitespace tokenization 
     2. Is your tokenization better when you don't allow merges across whitespace?
   What do you gain?  What do you lose?
 * Can you improve the efficiency (e.g., in searching or in merging)
 * Our default `min_frequency` was 2, is that the right choice?

The first step is to train your guesser:

    jbg@MacBook-Pro-von-Jordan GPT3QA % python3 guesser.py --guesser_type=ToyTokenizer --limit=10000 --questions=../data/qanta.guesstrain.json.gz
    Setting up logging
    INFO:root:Using device 'cpu' (cuda flag=False)
    INFO:root:Initializing guesser of type ToyTfidf
    INFO:root:Loading questions from ../data/qanta.guesstrain.json.gz
    INFO:root:Read 10000 questions
    100%|███████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 25450.41it/s]
    100%|███████████████████████████████████████████████████████████████████████████████████████████████| 6616/6616 [00:00<00:00, 1768498.84it/s]
    Creating vocabulary: 100%|██████████████████████████████████████████████████████████████████████████| 47648/47648 [00:04<00:00, 10791.36it/s]
    Creating document freq: 100%|████████████████████████████████████████████████████████████████████████| 47648/47648 [00:04<00:00, 9954.71it/s]
    Creating document vecs: 100%|████████████████████████████████████████████████████████████████████████| 47648/47648 [00:05<00:00, 8766.74it/s]

This will store a model in the model directory (you might need to
create it if you get an error).  Pay attention to the `limit`
argument, because that's going to save you a lot of time on real data.

Then see how it's doing on an evaluation set:

    jbg@MacBook-Pro-von-Jordan GPT3QA % python3 eval.py --evaluate=guesser --guesser_type='ToyTfidf' --questions=../data/qanta.guessdev.json.gz --limit=100 --load=True --num_guesses=1
    Setting up logging
    INFO:root:Loading questions from ../data/qanta.guessdev.json.gz
    INFO:root:Read 100 questions
    INFO:root:Using device 'cpu' (cuda flag=False)
    INFO:root:Initializing guesser of type ToyTfidf
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 411609.81it/s]
    INFO:root:Generating guesses for 100 new question
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [01:51<00:00,  1.11s/it]
    miss 0.97
    ===================
    
                   guess: Ernest_Hemingway
                  answer: Gerard_Manley_Hopkins
                      id: 93155
                    text: This author describes the title event of one poem as a "heart's
                          clarion" and "world's wildfire" and states that "this... poor
                          potsherd, patch, matchwoodÂ… is Immortal Diamond." This author wrote a
                          poem whose various interpretations center on the word "Buckle" and
                          describes how "blue-bleak embersÂ… Fall, gall themselves, and gash
                          gold-vermillion." This author of "That Nature is a Heraclitean Fire
                          and of the Comfort of the Resurrection" remembered "the Happy Memory
                          of five Franciscan Nuns exiled by the Falk Laws," in a poem that plays
                          on the name of a sunken ship to describe both the vessel and the
                          country from which it departed. For 10 points, name this English
                          Jesuit poet of "The Wreck of the Deutschland," who used sprung rhythm
                          in "The Windhover."
    --------------------
                   guess: Hydrochloric_acid
                  answer: Scarring
                      id: 93341
                    text: This process occurs in the tunica albuginea around the corpora
                          cavernosa in Peyronie's disease. Myofibroblasts contribute to this
                          process when they fail to disappear from granulation tissue via
                          apoptosis. In this process, proteins are oriented parallel to each
                          other rather than being oriented perpendicular to each other in the
                          proper "basket-weave" manner. This process can produce both
                          "hypertrophic" products and keloids. It's not inflammation, but the
                          muscle of the heart undergoes this process after a myocardial
                          infarction, and the tissues of the liver undergo this process in
                          patients with cirrhosis. For 10 points, name this process in which
                          excess connective tissue accumulates in response to events like
                          injuries to the skin.
    --------------------
                   guess: Canada
                  answer: North_Macedonia
                      id: 93194
                    text: This country's Titov Veles district is known for its high quality
                          opium. A campaign to build nationalist monuments in this country is
                          known as antiquisation. The Golem Grad, home to ruined churches and
                          thousands of snakes, can be found in this country's majority portion
                          of Lake Prespa. Using Motorola's Canopy technology, this country was
                          the first to achieve nationwide wireless broadband. In 1995, this
                          country was forced to remove a 16-rayed sun from its flag, as part of
                          a dispute that still keeps it out of the EU. This country was forced
                          to use a name abbreviated FYROM when it joined the UN. Lake Ohrid lies
                          on this country's border with Albania. For 10 points, name this
                          country that disputes a national identity with Greece, a former
                          Yugoslav republic with capital at Skopje.
    --------------------
                   guess: Federico_García_Lorca
                  answer: Henry_Wadsworth_Longfellow
                      id: 93253
                    text: The speaker of a poem by this author pledges, "I will keep you there
                          forever" in the "round-tower of my heart" "till the walls should
                          crumble to ruin, and molder in dust away." A poem by this author
                          begins by describing the farmer of Grand-Pre, who dies of a heart
                          attack after a mini-riot is quelled by Father Felician. Another of
                          this author's characters fasts for seven days, during which he
                          wrestles Mondamin. This poet of "The Children's Hour" wrote a poem in
                          dactylic hexameter beginning "This is the forest primeval," and used a
                          Kalevala-esque trochaic tetrameter for a poem in which the grandson of
                          Nokomis weds Minnehaha "By the shores of Gitche Gumee." For 10 points,
                          name this American poet of "Evangeline" who tried to draw on Ojibwe
                          lore in "The Song of Hiawatha."
    --------------------
                   guess: Taylor_Swift
                  answer: Lana_Del_Rey
                      id: 93140
                    text: This singer instructs "put me onto your black motorcycle" and asks to
                          "let me put on a show for you, daddy" in "Yayo," and repeats "go, go,
                          go, go, go, this is my show" in a song on her latest album. This
                          singer opens a song with "I've been out on that open road" before
                          describing how she "hears the birds on the summer breeze." In another
                          song, she invites you to "come on take a walk on the wild side" and
                          sings "the road is long, we carry on, try to have fun in the
                          meantime." This singer of "Ride" asks "will you still love me when I
                          got nothing but my aching soul?" in another song. Cedric Gervais
                          remixed her most successful song, in which she sings "kiss me hard
                          before you go." For 10 points, name this American singer of "Born to
                          Die," "Young and Beautiful," and "Summertime Sadness."
    --------------------
                   guess: South_Dakota
                  answer: State_of_Washington
                      id: 93174
                    text: This eventual state's first territorial governor, Isaac Stevens,
                          presided over settlers led by half-black pioneer George Washington
                          Bush. An aviator coincidentally named Harry Truman refused to evacuate
                          this home state of the Yakama tribe. A Supreme Court case originating
                          in this state overturned Adkins v. Children's Hospital via Owen
                          Roberts's "switch in time" saving a minimum-wage law. Plutonium for
                          the Fat Man bomb was produced at this state's Hanford site. Police in
                          this state attacked splinter groups such as DAN, who formed an
                          anarchist "black bloc" at 1999 protests against the World Trade
                          Organization. Elsie Parrish sued the West Coast Hotel in this state,
                          where Spirit Lake was destroyed in a 1980 disaster. For 10 points,
                          name this state where Mount St. Helens erupted.
    --------------------
                   guess: Cell_wall
                  answer: None
                      id: 93180
                    text: Transport between these two structures requires the BET1 membrane
                          protein. The GTPase SAR1A helps assemble Sec23p/24p and Sec13p/31p
                          into the coat that covers molecules being transported between these
                          structures. Transport between these structures is mediated by the
                          vesicular-tubular cluster, also called their namesake "intermediate
                          compartment." The presence of a KDEL sequence causes the continuous
                          retrieval of molecules from one of these organelles to the other. In
                          the first of these organelles, PDI creates disulfide bridges in
                          substrates which the second of these organelles might label with
                          mannose-6-phosphate. COPII coats proteins transported between these
                          organelles. For 10 points, name these two organelles, the first of
                          which folds proteins which are then packaged by the second.
    --------------------
                   guess: Solubility
                  answer: Inflation_(cosmology)
                      id: 93164
                    text: The spectral index n-sub-s is one parameter in this theory that
                          quantifies its departure from scale invariance. Early versions of this
                          theory involved false vacuum states that failed to account for the
                          radiation needed for reheating. Newer models use a scalar field with a
                          namesake parameter involving the ratio of the second time derivative
                          of the Hubble parameter to the product of the Hubble parameter and its
                          first time derivative: the "slow roll" parameter. This theory answers
                          the question of why the universe is isotropic if different areas are
                          not in causal contact, known as the horizon problem. The main process
                          this theory predicts was finished by 10 to the negative 32 seconds
                          after the Big Bang. Alan Guth developed, for 10 points, what theory
                          which posits a rapid expansion of the early universe?
    --------------------
                   guess: Antimony
                  answer: Silicon
                      id: 93284
                    text: When alpha to a sulfoxide group, groups containing this element
                          migrate in a common variation of the Pummerer rearrangement.
                          Transmetalation of groups from this element to palladium is a key step
                          in the Hiyama cross-coupling. When groups containing this element are
                          geminal or vicinal to hydroxyl groups, they rearrange to ethers of
                          this element in the Brook rearrangement. Enol derivatives of those
                          ethers of this element are reacted with aldehydes or formates in the
                          Mukaiyama aldol reaction, and ethers of this element are generally
                          useful protecting groups for alcohols. The tetramethyl derivative of
                          this element is defined to have a chemical shift of 0 ppm in proton
                          NMR, and its dioxide is the main component of glass. For 10 points,
                          name this tetravalent element used in microchips.
    --------------------
                   guess: Pablo_Neruda
                  answer: Twenty_Love_Poems_and_a_Song_of_Despair
                      id: 93290
                    text: The speaker of one poem in this collection describes himself as "the
                          word without echoes, he who lost everything and he who had everything"
                          after addressing "you who are silent," a white bee "drunk with honey"
                          that buzzes in the speaker's soul. This collection contains a poem
                          that includes the lines "The night is starry and the stars are blue
                          and shiver in the distance" and "Love is so short, forgetting is so
                          long." The speaker declares, "You look like a world lying in
                          surrender" after noting the "white hills, white thighs" of the title
                          thing in "Body of a Woman." The speaker of the last poem in this
                          collection repeatedly exclaims "In you everything sank!" right after a
                          poem beginning "Tonight I can write the saddest lines." For 10 points,
                          name this early poetry collection by Pablo Neruda.
    --------------------
    =================
    close 0.03
    ===================
    
    =================
    hit 0.03
    ===================
    
                   guess: Permian
                  answer: Permian
                      id: 93291
                    text: A sedimentary basin named for this period in western Texas has the
                          world's thickest deposits dating from here. Its latter part involved
                          the upward rifting of the Cimmerian subcontinent, an event that formed
                          the Neo-Tethys sea. Swamp-loving lycopod trees were gradually replaced
                          in continental interiors by advanced species of seed ferns and
                          conifers in this period. Olson's extinction is a small event that
                          occurred is in the middle of this period. By the end of this period,
                          dicynodonts and gorgonopsians dominated terrestrial fauna. The
                          supercontinent Pangea existed throughout this period, which the
                          Siberian traps may have ended. The Carboniferous period preceded, for
                          10 points, what last geologic period of the Paleozoic, which ended
                          with a massive extinction event that ushered in the Triassic?
    --------------------
                   guess: Pareto_efficiency
                  answer: Pareto_efficiency
                      id: 93321
                    text: The dual form of this concept was introduced by David Luenberger. In
                          incomplete markets, a constrained version of this concept is used
                          instead. The Scitovsky paradox results when this concept is extended
                          using a compensation principle. In an Edgeworth Box, this condition
                          occurs when the marginal rates of substitution are identical, which
                          happens along a contract curve. This condition is equivalent to
                          Walrasian equilibrium according to the fundamental theorems of welfare
                          economics. Kaldor and Hicks extended this concept to situations where
                          a redistribution of wealth could result in greater total utility. For
                          10 points, name this term for an allocation in which no person can be
                          made better off without making another worse off, named for an Italian
                          economist.
    --------------------
                   guess: Augustin-Louis_Cauchy
                  answer: Augustin-Louis_Cauchy
                      id: 93323
                    text: Terms with indices that are powers of two are used in this
                          mathematician's namesake condensation test to check for convergence of
                          a series. With Hadamard, this mathematician names a formula that
                          calculates the radius of convergence of a power series. He's not
                          Lagrange, but he names a theorem stating that if a prime p divides the
                          order of a group, then the group contains an element with order p.
                          Partial differential equations that give a necessary and sufficient
                          condition for a complex function to be holomorphic are named for him
                          and Riemann. Sequences named for this man, by definition, converge
                          inside complete metric spaces. For 10 points, name this French
                          mathematician who names an inequality relating the dot product of two
                          vectors and their magnitudes along with Hermann Schwarz.
    --------------------
    =================
    Precision @1: 0.0300 Recall: 0.0300

Extra Credit (5 Points)
=================

Make your code go faster for both training and testing.  One way to do
this is to override the `batch_guess` function.

You can also get extra credit for submitting new test cases that test
functionality that the supplied tests miss (i.e., if you discover a
bug that isn't covered by our tests, you can get extra credit by
submitting a test that covers it).

# Submission Instructions


1. Submissions will be made on Gradescope.

2. You will submit a zip file containing your code
(toyttokenizer_guesser.py) and the model files (if you go beyond the "good
enough" standard). If you check your code against your own test cases,
you can add the file (which will be like test.py) containing your own
test cases in the zip too.

The code will run against the public test cases (the ones you can already see
in the given test.py file) on the server and you can see those results. You
should make sure you pass these cases before the submission deadline.

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

**Q: Why are we computing `global_freq`, it doesn't seem to be needed
for tf-df.**

**A:** That's right.  But it's easy to write a unit test for, and the
logic for computing it will help you compute the document frequencies.

**Q: In the unit tests, how do we have one token that has frequency
2/3 and two different tokens that have frequency 1/3?**

**A:** They're not really two different tokens, they both got mapped
to the unknown token.

**Q: Why is there a ``page`` field and an ``answer`` field.  Which one do I use?**

**A:** As you can see, the ``answer`` field has inconsistent formatting and is sometimes ambiguious.   To make things a little more sane, we map all of the answers to Wikipedia page titles.  This makes it so that rather than having to guess exactly the crazy formatting of the answer line, the Guesser just needs to match up to the correct underlying entity.  Not all questions have pages, which does cause a problem, but we're going to ignore that issue for a while, as most of them do have pages associated.


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

In other words, we'll be building a matrix `_doc_vectors` in the
code.  It will have rows corresponding to the number of documents and
columns corresponding to the number of words.  But at first we don't
know how many words or documents we have, so we'll do this in three
steps:
 * Find out how many words we have
 * Find out how many documents we have
 * Build the matrix

What's the Vocab
---------------

Now, we could do this more efficiently without three passes, but this
will make debugging simpler.  After the first two steps, to let the
code know that we're done, we'll call the `finalize_vocab` and
`finalize_docs` functions to tell the code to not let that change any
more.  After that point, we won't be able to tell the code that we've
seen new words or new documents.

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
become the "unknown token" (`kUNK` in the code).
 
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

What are the Documents?
---------------------

Okay, so that's our vocabulary.  We also need to compute statistics
for tf-idf.  We can't do everything at once, so we'll need to do two
passes over the data.  The first pass will count how many times we see
each word in the training set (using the function ``train_seen``), and
the second pass will compute term and document frequencies (using the
function ``add_document``).  In between those two passes, we'll
finalize our vocabulary to decide the integer lookup of all of our
words (the ``finalize_vocab`` function).

Then, you should have everything you need to compute---for a new
document or query---the tf-idf representation in the ``doc_tfidf``
function!

What to Do
=============

# Code (20 Points)

You'll need to complete several functions in the ToyTfIdf_Guesser class:
* `constructor`
* `vocab_seen`
* `finalize_vocab`
* `scan_document`
* `global_freq`
* `inv_docfreq`
* `__call__`

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

The only trick here is that for going beyond "good enough", everything
that needs to run the idf calculation needs to get saved in the `save`
function.  So if you add anything beyond `_doc_counts` here, you're
going to neeed to save it in the `save` function.

vocab_seen
----------

Here, you'll need to take the *string* representations that you've
seen and keep track of how often you've seen them.

finalize_vocab
----------

Once we've done a scan over all of the documents, we can create a
vocabulary, taking all of the words that have appeared more than or
equal to unk_cutoff times into the vocabulary.  You make want to free
up the memory you used for train_seen here, but not necessary for
getting a good grade.

The most important thing is that after this function has run, the
vocab data member of TfIdf should map words (in string form) to their
vocabulary id (an integer).

scan_document
---------

Take in a document and keep track of the words that you see in
appropriate datastructures so that the next two functions work.

global_freq
----------

Return the frequency of a word.

inv_docfreq
-------------

Return the inverse document frequency of a word.

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


Running Your Code
=================

First, make sure you pass the unit tests.  Inititally, it will look
like this:

    jbg@MacBook-Pro-von-Jordan tfidf % python3 toytfidf_test.py
    /Users/jbg/repositories/nlp-hw/tfidf/toytfidf_guesser.py:228: DeprecationWarning: The 'warn' function is deprecated, use 'warning' instead
      logging.warn("Vocab size is very small, this suggests either you didn't implement vocabulary, the dataset is small, or your filters are too aggressive")
    WARNING:root:Vocab size is very small, this suggests either you didn't implement vocabulary, the dataset is small, or your filters are too aggressive
    ['0', '0', '0']
    .WARNING:root:Vocab size is very small, this suggests either you didn't implement vocabulary, the dataset is small, or your filters are too aggressive
    FWARNING:root:Vocab size is very small, this suggests either you didn't implement vocabulary, the dataset is small, or your filters are too aggressive
    FWARNING:root:Vocab size is very small, this suggests either you didn't implement vocabulary, the dataset is small, or your filters are too aggressive
    F
    ======================================================================
    FAIL: test_df (__main__.TestSequenceFunctions.test_df)
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "/Users/jbg/repositories/nlp-hw/tfidf/toytfidf_test.py", line 87, in test_df
        self.assertAlmostEqual(self.guesser.inv_docfreq(word_a), log10(1.3333333))
    AssertionError: 0.0 != 0.12493872575093778 within 7 places (0.12493872575093778 difference)
    
    ======================================================================
    FAIL: test_tf (__main__.TestSequenceFunctions.test_tf)
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "/Users/jbg/repositories/nlp-hw/tfidf/toytfidf_test.py", line 65, in test_tf
        self.assertAlmostEqual(self.guesser.global_freq(word_a), 0.66666666)
    AssertionError: 0.0 != 0.66666666 within 7 places (0.66666666 difference)
    
    ======================================================================
    FAIL: test_vocab (__main__.TestSequenceFunctions.test_vocab)
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "/Users/jbg/repositories/nlp-hw/tfidf/toytfidf_test.py", line 30, in test_vocab
        self.assertNotEqual(self.guesser.vocab_lookup("a"),
    AssertionError: 0 == 0
    
    ----------------------------------------------------------------------
    Ran 4 tests in 0.001s
    
    FAILED (failures=3)

After all of the tests have been passed, it will look like this:

    jbg@MacBook-Pro-von-Jordan GPT3QA % python3 toytfidf_test.py
    ['0', '1', '1']
    ....
    ----------------------------------------------------------------------
    Ran 4 tests in 0.000s
    
    OK

However, these tests aren't very realistic.  You will also want to run
the code with some small English data that we've provided.  You can do
this via the main method of the `toytfidf_guesser.py` file:

    jbg@MacBook-Pro-von-Jordan tfidf % python3 toytfidf_guesser.py --guesser_type=ToyTfidf
    Setting up logging
    INFO:root:Using device 'cpu' (cuda flag=False)
    INFO:root:Initializing guesser of type ToyTfidf
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 91486.50it/s]
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 599186.29it/s]
    Creating vocabulary: 100%|████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 10626.77it/s]
    WARNING:root:Vocab size is very small, this suggests either you didn't implement vocabulary, the dataset is small, or your filters are too aggressive
    DEBUG:root:1 vocab elements, including: dict_keys(['<UNK>'])
    Creating document freq: 100%|█████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 13669.08it/s]
    DEBUG:root:Document counts final after 13 docs, some example inverse document frequencies:
    DEBUG:root:     <UNK> (  0): 0.00 0
    Creating document vecs: 100%|██████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 9187.19it/s]
    DEBUG:root:Document matrix is 13 by 1, has 0 non-zero entries
    ----------------------
    This capital of England [{'question': 'For 10 points, name this New England state with capital at Augusta.', 'guess': 'Maine', 'confidence': 0.0}]
    ----------------------
    The author of Pride and Prejudice [{'question': 'For 10 points, name this New England state with capital at Augusta.', 'guess': 'Maine', 'confidence': 0.0}]
    ----------------------
    The composer of the Magic Flute [{'question': 'For 10 points, name this New England state with capital at Augusta.', 'guess': 'Maine', 'confidence': 0.0}]
    ----------------------
    The economic law that says 'good money drives out bad' [{'question': 'For 10 points, name this New England state with capital at Augusta.', 'guess': 'Maine', 'confidence': 0.0}]
    ----------------------
    located outside Boston, the oldest University in the United States [{'question': 'For 10 points, name this New England state with capital at Augusta.', 'guess': 'Maine', 'confidence': 0.0}]

Because of the template code we gave you, this will always return the
first document as the answer (which is bad)!  Once you've done the
assignment, you'll get more reasonable answers:

    jbg@MacBook-Pro-von-Jordan tfidf % python3 toytfidf_guesser.py --guesser_type=ToyTfidf
    Setting up logging
    INFO:root:Using device 'cpu' (cuda flag=False)
    INFO:root:Initializing guesser of type ToyTfidf
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 92889.19it/s]
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 524288.00it/s]
    Creating vocabulary: 100%|█████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 9352.65it/s]
    DEBUG:root:42 vocab elements, including: dict_keys([',', 'this', '.', 'for', 'points', 'the', '10', '
    Creating document freq: 100%|█████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 12333.40it/s]
    DEBUG:root:Document counts final after 13 docs, some example inverse document frequencies:
    DEBUG:root:         , (  0): 0.07 0
    DEBUG:root:      this (  1): 0.03 0
    DEBUG:root:         . (  2): 0.03 0
    DEBUG:root:       for (  3): 0.16 0
    DEBUG:root:    points (  4): 0.16 0
    DEBUG:root:       the (  5): 0.34 0
    DEBUG:root:        10 (  6): 0.21 0
    DEBUG:root:      name (  7): 0.21 0
    DEBUG:root:        of (  8): 0.27 0
    DEBUG:root:        in (  9): 0.41 0
    Creating document vecs: 100%|██████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 7761.70it/s]
    DEBUG:root:Document matrix is 13 by 42, has 137 non-zero entries
    ----------------------
    This capital of England [{'question': 'For 10 points, name this city in New England, the capital of Massachusetts.', 'guess': 'Boston', 'confidence': 0.7283642385486387}]
    ----------------------
    The author of Pride and Prejudice [{'question': 'For 10 points, name this author of Pride and Prejudice.', 'guess': 'Jane_Austen', 'confidence': 0.9437740553090668}]
    ----------------------
    The composer of the Magic Flute [{'question': 'For 10 points, name this composer of Magic Flute and Don Giovanni.', 'guess': 'Wolfgang_Amadeus_Mozart', 'confidence': 0.7541277482065505}]
    ----------------------
    The economic law that says 'good money drives out bad' [{'question': 'FTP name this economic law which, in simplest terms, states that bad money drives out the good.', 'guess': "Gresham's_law", 'confidence': 0.8357018242587817}]
    ----------------------
    located outside Boston, the oldest University in the United States [{'question': 'It is the site of the National University of San Marcos, the oldest university in South America.', 'guess': 'Lima', 'confidence': 0.72061690198126}]

What's "Good Enough"?
==================
If you do all of this, you'll have a fine submission.  To go beyond
that, you'll need to save your code and have it run on the leaderboard
(not just pass the tests).  This means that you'll train a guesser
locally, upload the result, and then that will be scored on the
leaderboard.

The first step is to train your guesser:

    jbg@MacBook-Pro-von-Jordan GPT3QA % python3 guesser.py --guesser_type=ToyTfidf --limit=10000 --questions=../data/qanta.guesstrain.json.gz
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
(toytfidf_guesser.py) and the model files (if you go beyond the "good
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

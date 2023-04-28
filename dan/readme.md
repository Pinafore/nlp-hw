Deep Learning 
=

Overview
--------

To gain a better understanding of deep learning, we're going to look
at deep averaging networks (DAN).  These are a very simple framework,
but they work well for a variety of tasks and will help introduce some
of the core concepts of using deep learning in practice.

In this homework, you'll use Pytorch to implement a DAN model for
determining the answer to a Quizbowl question.

You'll turn in your code on Gradescope. This assignment is worth 20 points.

Dataset
----------------

We're working with the same data as before, except this time (because
we need to use representations) we will need to create a vocabulary
explicitly (like we did for the earlier tf-idf homework).  However,
we'll give you that code.  

Pytorch DataLoader
----------------

In this homework, we use Pytorch's build-in data loader to do data
mini-batching, which provides single or multi-process iterators over the
dataset(https://pytorch.org/docs/stable/data.html).

The data loader includes two functions, `batchify()` and `vectorize()`. For
each example, we need to vectorize the question text into a vector using the 
vocabulary. In this assignment, you need to write the `vectorize()` function
yourself. We provide the `batchify()` function to split the dataset into
mini-batches.

Guide
-----

First, you need to check to make sure that you can construct an example from
text.  This is called "vectorizing" in the Pytorch pipeline.

    > python3 dan_test.py 
    Traceback (most recent call last):
    ======================================================================
    FAIL: test_train_preprocessing (__main__.DanTest)
    On the toy data, make sure that create_indices creates the correct vocabulary and
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "/home/jbg/repositories/nlp-hw/dan/dan_test.py", line 155, in test_train_preprocessing
        self.assertEqual(guesser.vectorize(question), [3, 1])
    AssertionError: Lists differ: [0, 0] != [3, 1]

    First differing element 0:
    0
    3

    - [0, 0]
    + [3, 1]

Next, make sure that the network works correctly.  The unit tests define a
network that embeds the vocabulary and has two linear layers:

        embedding = [[ 0,  0],           # UNK
                     [ 1,  0],           # England
                     [-1,  0],           # Russia                     
                     [ 0,  1],           # capital
                     [ 0, -1],           # currency
                     ]

        first_layer = [[1, 0], [0, 1]] # Identity matrix
            
        second_layer = [[ 1,  1],        # -> London
                        [-1,  1],        # -> Moscow                        
                        [ 1, -1],        # -> Pound
                        [-1, -1],        # -> Rouble
                        ]

Those matrices are put into the parameters of the embeddings and linear layers:

        with torch.no_grad():
            self.toy_qa.linear1.bias *= 0.0
            self.toy_qa.linear2.bias *= 0.0
            self.toy_qa.embeddings.weight = nn.Parameter(torch.FloatTensor(embedding))
            self.toy_qa.linear1.weight.copy_(torch.FloatTensor(first_layer))
            self.toy_qa.linear2.weight.copy_(torch.FloatTensor(second_layer))

This should be a hint that you need to put these layers into a network of some sort!

After you've done that, the system should perfectly answer these questions
(e.g., that the "currency England" is the "Pound").  However, this is not the case at first:

    > python3 dan_test.py 
    Traceback (most recent call last):
      File "/home/jbg/repositories/nlp-hw/dan/dan_test.py", line 123, in testCorrectPrediction
        self.assertEqual(self.toy_dan_guesser.vectorize(words), indices)
    AssertionError: Lists differ: [0, 0] != [3, 1]

    First differing element 0:
    0
    3

    - [0, 0]
    + [3, 1]

Once you have things working, you'll need to train a network.

    python3 guesser.py --guesser_type=DanGuesser --question_source=gzjson --questions=../data/qanta.guesstrain.json.gz --secondary_questions=../data/qanta.guessdev.json.gz --limit=10000 --no_cuda


Then check to see how well the code does.

    > python3 eval.py --guesser_type=DanGuesser --question_source=gzjson --questions=../data/qanta.guessdev.json.gz --evaluate guesser --limit=250
    INFO:root:Generating guesses for 250 new question

    miss 0.69
    ===================
                   guess: Distillation
              answer: Lysis
                  id: 93198
                text: This process can be induced in cells by sodium deoxycholate or NP-40.
                      In another context, the Rz and Rz1 proteins help induce this process.
                      Gram-positive bacteria undergo this process when acted upon by an
                      enzyme present in hen egg white also known as muramidase. In protists,
                      this process is continually averted by the action of acidocalcisomes
                      in tandem with contractile vacuoles. This process occurs when a cell
                      is placed in an excessively hypotonic solution. A viral reproduction
                      cycle named for the fact that it causes the host cell to undergo this
                      process is contrasted with the lysogenic cycle. For 10 points, name
                      this general process in which a cell is destroyed via the rupturing of
                      its membrane.

    close 0.31
    ===================

               guess: Robert_Frost
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

    hit 0.02
    ===================

               guess: David_Hume
              answer: David_Hume
                  id: 93165
                text: The dominant edition of this philosopher's works was revised in the
                      70s by P.H. Nidditch and first edited by Baronet L. A. Selby-Bigge.
                      This man inspired "quasi-realism," a meta-ethical view formulated by
                      Simon Blackburn. Elizabeth Anscombe's Intention rejected this
                      thinker's two-component theory of motivation, in which a desire and a
                      belief are the only things needed to produce action. This man's
                      longest book claims that personal identity is a mere "bundle" of
                      perceptions, and downplays the strength of reason in its second part,
                      "Of Passions." This billiards enthusiast and secret atheist wrote that
                      all ideas come from prior "impressions," except perhaps a spectrum's
                      missing shade of blue. For 10 points, name this author of A Treatise
                      of Human Nature, an 18th-century empiricist Scotsman.

    =================
    Precision @1: 0.0200 Recall: 0.3080

Because many of you don't have GPUs, our goal is not to have you train a
super-converged model.  We want to see models with a non-zero recall and
precision guess at least hundreds of possible answers.  It doesn't have to be
particularly good (but you can get extra credit if you invest the time).


What you have to do
----------------

**Coding**: (15 points)
1. Understand the structure of the code.
2. Write the data `vectorize()` funtion.
3. Write DAN model initialization. 
4. Write model `forward()` function.
5. Write the model training/testing function. We don't have unit tests for this part, but it's necessary to get it correct to achieve reasonable performance.

**Analysis**: (5 points)
1. Report the accuracy on the dev set. 
2. Look at the development set and give some examples and explain the possible reasons why these examples are predicted incorrectly (remember that this is what eval.py does for you). 


Pytorch install
----------------
In this homework, we use Pytorch.  

You can install it via the following command (linux):
```
conda install pytorch torchvision -c pytorch
```

If you are using MacOS or Windows, please check the Pytorch website for installation instructions.

For more information, check
https://pytorch.org/get-started/locally/.

Extra Credit
----------------

For extra credit, you need to initialize the word representations with
word2vec, GloVe, or some other representation.  Compare the final performance
based on these initializations *and* see how the word representations
change. Write down your findings in analysis.pdf.

You can also get extra credit by getting the highest precision and recall by
tuning training parameters.

What to turn in 
----------------

TODO: Update for Gradescope

0. Submit your model file
1. Submit your `dan_guesser.py` file.
2. Submit your `analysis.pdf` file. (Please make sure that this is **PDF** file!      No more than one page, include your name at the top of the pdf.)
3. Upload your model parameters.
4. (Optional) Upload the wordvectors you use.

FAQ
----

*Q:* There aren't enough answers or too many!  What can I do?

*A:* Look at the DanGuesser_min_answer_freq flag to adjust what answers you include.

*Q:* Too many of the answers are unknown!  What can I do?

*A:* Look at the DanGuesser_unk_drop flag to adjust how many "unknown" examples you keep.

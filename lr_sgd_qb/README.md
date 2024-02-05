Stochastic Gradient Descent
=

Overview
--------

In this homework you'll implement a stochastic gradient ascent for logistic
regression and you'll apply it to the task of determining whether an answer to
a question is correct or not.  This is the **buzzer** task that we'll be using
in subsequent homeworks, so play attention!

This will be slightly more difficult than the last homework (the difficulty
will slowly ramp upward).  You should not use any libraries that implement any
of the functionality of logistic regression for this assignment; logistic
regression is implemented in scikit learn, pytorch, and many other places, but
you should do everything by hand now.  You'll be able to use library
implementations of logistic regression in the future (and add your own features).

You'll turn in your code on Gradescope.  This assignment is worth 30 points.

The Data
---
As always, start with the unit tests.  That uses the same toy data we used in the workshop to explain what's going on.  But eventually you'll graduate to [real data](https://github.com/Pinafore/nlp-hw/blob/master/data/small_guess.buzztrain.jsonl).

     {"guess:The Soldier (play)": 1, "Gpr_confidence": -0.71123162386, "Length_char": -0.7755555555555556, "Length_word": -0.7733333333333333, "Length_ftp": 0, "Length_guess": 2.9444389791664403, "Frequency_guess": 0.0, "Category_category:Literature": 1, "Category_year": 3.5553480614894135, "Category_subcategory:Literature Classical": 1, "Category_tournament:ACF Regionals": 1, "label": false}
     {"guess:Hamlet": 1, "Gpr_confidence": -1.3516115696, "Length_char": -0.5488888888888889, "Length_word": -0.5333333333333333, "Length_ftp": 0, "Length_guess": 1.9459101490553132, "Frequency_guess": 3.5553480614894135, "Category_category:Literature": 1, "Category_year": 3.5553480614894135, "Category_subcategory:Literature Classical": 1, "Category_tournament:ACF Regionals": 1, "label": false}
     {"guess:Hamlet": 1, "Gpr_confidence": -0.7734369171800001, "Length_char": -0.33111111111111113, "Length_word": -0.26666666666666666, "Length_ftp": 0, "Length_guess": 1.9459101490553132, "Frequency_guess": 3.5553480614894135, "Category_category:Literature": 1, "Category_year": 3.5553480614894135, "Category_subcategory:Literature Classical": 1, "Category_tournament:ACF Regionals": 1, "label": false}
     {"guess:Timon of Athens": 1, "Gpr_confidence": -0.29131036656750003, "Length_char": -0.10888888888888888, "Length_word": -0.013333333333333334, "Length_ftp": 0, "Length_guess": 2.772588722239781, "Frequency_guess": 2.1972245773362196, "Category_category:Literature": 1, "Category_year": 3.5553480614894135, "Category_subcategory:Literature Classical": 1, "Category_tournament:ACF Regionals": 1, "label": false}
{"guess:Timon of Athens": 1, "Gpr_confidence": -0.5494337382, "Length_char": 0.1111111111111111, "Length_word": 0.21333333333333335, "Length_ftp": 0, "Length_guess": 2.772588722239781, "Frequency_guess": 2.1972245773362196, "Category_category:Literature": 1, "Category_year": 3.5553480614894135, "Category_subcategory:Literature Classical": 1, "Category_tournament:ACF Regionals": 1, "label": false}
     {"guess:Mark Antony": 1, "Gpr_confidence": -0.33353722919999995, "Length_char": 0.34, "Length_word": 0.4266666666666667, "Length_ftp": 0, "Length_guess": 2.4849066497880004, "Frequency_guess": 2.9444389791664403, "Category_category:Literature": 1, "Category_year": 3.5553480614894135, "Category_subcategory:Literature Classical": 1, "Category_tournament:ACF Regionals": 1, "label": true}
{"guess:Mark Antony": 1, "Gpr_confidence": -0.501373298, "Length_char": 0.5666666666666667, "Length_word": 0.6533333333333333, "Length_ftp": 1, "Length_guess": 2.4849066497880004, "Frequency_guess": 2.9444389791664403, "Category_category:Literature": 1, "Category_year": 3.5553480614894135, "Category_subcategory:Literature Classical": 1, "Category_tournament:ACF Regionals": 1, "label": true}
     {"guess:Mark Antony": 1, "Gpr_confidence": -0.00858155044, "Length_char": 0.7755555555555556, "Length_word": 0.84, "Length_ftp": 1, "Length_guess": 2.4849066497880004, "Frequency_guess": 2.9444389791664403, "Category_category:Literature": 1, "Category_year": 3.5553480614894135, "Category_subcategory:Literature Classical": 1, "Category_tournament:ACF Regionals": 1, "label": true}

These are guesses generated to a question whose answer is *Mark Antony*.  It only gets it right near the end of the question.  Previously, it was guessing *Timon of Athens* and *Hamlet* instead.  Our goal is to predict when the guess was correct.  There are a couple of features here that can help it detect when it is right:

  * `Gpr_confidence`: The average log probability of the generated answer tokens
  * `Length_char`: How long the question is in terms of characters
  * `Length_word`: How long the question is in terms of words
  * `Category_category` / `Category_subcategory`: What the question is about; there are also features for which tournament it came from
  * `guess`: What the guesser guessed 
  * `label`: **NOT A FEATURE** This is what is predicted: whether the question was right or not

When you actually run your code, many of these features will disappear because they're not in the [vocab](https://github.com/Pinafore/nlp-hw/blob/master/data/small_guess.vocab) (I thought they'd work better than they did).  So to make things as simple as possible, the vocab just includes the things that seemed to work well.  You're welcome to try adding things back in, though!

What you have to do
----

Coding (25 points):

1. Understand how the code is creating feature vectors (this will help you
code the solution and to do the later analysis).  You don't actually need to
write any code for this, however.  

2. (Optional) Store necessary data in the constructor so you can do
classification later.

3. Modify the _sg update_ function to perform updates.

4. Modify the _inspect_ function to return the most salient features

Analysis (5 points):

1. What is the role of the learning rate?
2. How many datapoints (or multiple passes over the data) do you need to
complete for the *model* to stabilize?  The various metrics can give you clues
about what the model is doing, but no one metric is perfect.
3. What do the features tell you about the underlying problem?

Extra credit:

1.  Modify the _sg update_ function to perform [lazy regularized updates](https://lingpipe.files.wordpress.com/2008/04/lazysgdregression.pdf), which only update the weights of features when they appear in an example.
    - Show the effect in your analysis document 
    
Caution: When implementing extra credit, make sure your implementation of the
regular algorithms doesn't change.

What's good enough?
-

You do not need to tune the regularization / learning rate to get a great
accuracy / precision / recall.  The main requirements are to get the math of
the updates correct and to understand what's going on.  However, if your
accuracy is not above 0.6 or if your precision or recall is zero, there is a
problem.


What to turn in
-

1. Submit your _lr_sgd.py_ file (include your name at the top of the source)
1. Submit your _analysis.pdf_ file
    - no more than one page (NB: This is also for the extra credit.  To minimize effort for the grader, you'll need to put everything on a page.  Take this into account when selecting if/which extra credit to do...think of the page requirement like a regularizer).
    - pictures are better than text
    - include your name at the top of the PDF

Unit Tests
=

I've provided unit tests based on the [example](
https://users.umiacs.umd.edu/~jbg/teaching/CMSC_470/04b_ex.pdf) that we worked
through in class.  Before running your code on read data, make sure it passes
all of the unit tests.

```
cs244-33-dhcp:logreg jbg$ python tests.py
.[ 0.  0.  0.  0.  0.]
[ 1.  4.  3.  1.  0.]
F
======================================================================
FAIL: test_unreg (__main__.TestKnn)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests.py", line 22, in test_unreg
    self.assertAlmostEqual(b1[0], .5)
AssertionError: 0.0 != 0.5 within 7 places

----------------------------------------------------------------------
Ran 2 tests in 0.001s

FAILED (failures=1)
```

Example
-

This is an example of what your runs should look like:
```
.venv/bin/python3 toylogistic_buzzer.py --train=data/small_guess.buzztrain.jsonl --test=data/small_guess.buzzdev.jsonl --vocab=data/small_guess.vocab --passes=1 --learning_rate=0.01 --regularization=0.1
Loaded 16 items from vocab data/small_guess.vocab
Read in 6619 train and 1628 test
INFO:root:Creating regression over 16 features
INFO:root:Update      1	Trainlogprob = -4572.544	Trainacc = 0.569	Trainprec = 0.569	Trainrecall = 1.000	Devlogprob = -1130.929	Devacc = 0.485	Devprec = 0.485	Devrecall = 1.000	
INFO:root:Update    101	Trainlogprob = -4230.731	Trainacc = 0.570	Trainprec = 0.569	Trainrecall = 1.000	Devlogprob = -1083.298	Devacc = 0.489	Devprec = 0.487	Devrecall = 1.000	
INFO:root:Update    201	Trainlogprob = -3909.551	Trainacc = 0.730	Trainprec = 0.696	Trainrecall = 0.931	Devlogprob = -980.354	Devacc = 0.720	Devprec = 0.641	Devrecall = 0.959	
INFO:root:Update    301	Trainlogprob = -3817.780	Trainacc = 0.738	Trainprec = 0.710	Trainrecall = 0.912	Devlogprob = -954.137	Devacc = 0.735	Devprec = 0.657	Devrecall = 0.952	
INFO:root:Update    401	Trainlogprob = -3781.884	Trainacc = 0.738	Trainprec = 0.745	Trainrecall = 0.819	Devlogprob = -912.750	Devacc = 0.770	Devprec = 0.713	Devrecall = 0.882	
INFO:root:Update    501	Trainlogprob = -3838.177	Trainacc = 0.738	Trainprec = 0.711	Trainrecall = 0.911	Devlogprob = -950.156	Devacc = 0.736	Devprec = 0.660	Devrecall = 0.944	
INFO:root:Update    601	Trainlogprob = -3788.682	Trainacc = 0.727	Trainprec = 0.690	Trainrecall = 0.942	Devlogprob = -954.522	Devacc = 0.713	Devprec = 0.634	Devrecall = 0.965	
INFO:root:Update    701	Trainlogprob = -3841.041	Trainacc = 0.731	Trainprec = 0.697	Trainrecall = 0.933	Devlogprob = -959.140	Devacc = 0.724	Devprec = 0.644	Devrecall = 0.963	
INFO:root:Update    801	Trainlogprob = -3764.710	Trainacc = 0.741	Trainprec = 0.762	Trainrecall = 0.791	Devlogprob = -896.787	Devacc = 0.770	Devprec = 0.726	Devrecall = 0.847	
INFO:root:Update    901	Trainlogprob = -3712.051	Trainacc = 0.745	Trainprec = 0.727	Trainrecall = 0.883	Devlogprob = -910.584	Devacc = 0.755	Devprec = 0.683	Devrecall = 0.923	
INFO:root:Update   1001	Trainlogprob = -3745.356	Trainacc = 0.746	Trainprec = 0.735	Trainrecall = 0.864	Devlogprob = -911.474	Devacc = 0.760	Devprec = 0.693	Devrecall = 0.910	
INFO:root:Update   1101	Trainlogprob = -3773.589	Trainacc = 0.739	Trainprec = 0.765	Trainrecall = 0.782	Devlogprob = -897.185	Devacc = 0.771	Devprec = 0.732	Devrecall = 0.833	
INFO:root:Update   1201	Trainlogprob = -3686.121	Trainacc = 0.738	Trainprec = 0.709	Trainrecall = 0.914	Devlogprob = -916.504	Devacc = 0.738	Devprec = 0.660	Devrecall = 0.949	
INFO:root:Update   1301	Trainlogprob = -3702.419	Trainacc = 0.742	Trainprec = 0.719	Trainrecall = 0.898	Devlogprob = -916.572	Devacc = 0.750	Devprec = 0.674	Devrecall = 0.939	
INFO:root:Update   1401	Trainlogprob = -3731.359	Trainacc = 0.739	Trainprec = 0.757	Trainrecall = 0.796	Devlogprob = -890.430	Devacc = 0.770	Devprec = 0.723	Devrecall = 0.854	
INFO:root:Update   1501	Trainlogprob = -3705.625	Trainacc = 0.742	Trainprec = 0.737	Trainrecall = 0.847	Devlogprob = -899.691	Devacc = 0.766	Devprec = 0.702	Devrecall = 0.900	
INFO:root:Update   1601	Trainlogprob = -3700.467	Trainacc = 0.741	Trainprec = 0.726	Trainrecall = 0.874	Devlogprob = -909.501	Devacc = 0.756	Devprec = 0.683	Devrecall = 0.927	
INFO:root:Update   1701	Trainlogprob = -3828.270	Trainacc = 0.728	Trainprec = 0.778	Trainrecall = 0.729	Devlogprob = -901.437	Devacc = 0.768	Devprec = 0.754	Devrecall = 0.775	
INFO:root:Update   1801	Trainlogprob = -3700.722	Trainacc = 0.746	Trainprec = 0.747	Trainrecall = 0.839	Devlogprob = -890.751	Devacc = 0.767	Devprec = 0.705	Devrecall = 0.894	
INFO:root:Update   1901	Trainlogprob = -3646.731	Trainacc = 0.745	Trainprec = 0.744	Trainrecall = 0.839	Devlogprob = -876.589	Devacc = 0.764	Devprec = 0.701	Devrecall = 0.894	
INFO:root:Update   2001	Trainlogprob = -3627.108	Trainacc = 0.740	Trainprec = 0.749	Trainrecall = 0.816	Devlogprob = -865.009	Devacc = 0.773	Devprec = 0.717	Devrecall = 0.878	
INFO:root:Update   2101	Trainlogprob = -3716.854	Trainacc = 0.733	Trainprec = 0.699	Trainrecall = 0.931	Devlogprob = -936.431	Devacc = 0.721	Devprec = 0.642	Devrecall = 0.959	
INFO:root:Update   2201	Trainlogprob = -3698.057	Trainacc = 0.745	Trainprec = 0.735	Trainrecall = 0.863	Devlogprob = -900.957	Devacc = 0.761	Devprec = 0.693	Devrecall = 0.911	
INFO:root:Update   2301	Trainlogprob = -3707.811	Trainacc = 0.736	Trainprec = 0.765	Trainrecall = 0.773	Devlogprob = -874.597	Devacc = 0.771	Devprec = 0.735	Devrecall = 0.825	
INFO:root:Update   2401	Trainlogprob = -3670.291	Trainacc = 0.742	Trainprec = 0.728	Trainrecall = 0.872	Devlogprob = -900.824	Devacc = 0.756	Devprec = 0.684	Devrecall = 0.922	
INFO:root:Update   2501	Trainlogprob = -3705.221	Trainacc = 0.734	Trainprec = 0.756	Trainrecall = 0.786	Devlogprob = -880.578	Devacc = 0.772	Devprec = 0.727	Devrecall = 0.849	
INFO:root:Update   2601	Trainlogprob = -3695.008	Trainacc = 0.740	Trainprec = 0.751	Trainrecall = 0.811	Devlogprob = -884.586	Devacc = 0.770	Devprec = 0.716	Devrecall = 0.873	
INFO:root:Update   2701	Trainlogprob = -3694.506	Trainacc = 0.741	Trainprec = 0.716	Trainrecall = 0.902	Devlogprob = -917.666	Devacc = 0.746	Devprec = 0.669	Devrecall = 0.944	
INFO:root:Update   2801	Trainlogprob = -3717.630	Trainacc = 0.735	Trainprec = 0.706	Trainrecall = 0.917	Devlogprob = -930.532	Devacc = 0.733	Devprec = 0.654	Devrecall = 0.952	
INFO:root:Update   2901	Trainlogprob = -3732.705	Trainacc = 0.737	Trainprec = 0.706	Trainrecall = 0.921	Devlogprob = -936.084	Devacc = 0.733	Devprec = 0.654	Devrecall = 0.956	
INFO:root:Update   3001	Trainlogprob = -3750.907	Trainacc = 0.733	Trainprec = 0.700	Trainrecall = 0.928	Devlogprob = -942.993	Devacc = 0.727	Devprec = 0.648	Devrecall = 0.957	
INFO:root:Update   3101	Trainlogprob = -3715.802	Trainacc = 0.743	Trainprec = 0.721	Trainrecall = 0.896	Devlogprob = -917.605	Devacc = 0.746	Devprec = 0.670	Devrecall = 0.938	
INFO:root:Update   3201	Trainlogprob = -3727.140	Trainacc = 0.741	Trainprec = 0.717	Trainrecall = 0.900	Devlogprob = -924.757	Devacc = 0.744	Devprec = 0.668	Devrecall = 0.939	
INFO:root:Update   3301	Trainlogprob = -3707.754	Trainacc = 0.740	Trainprec = 0.716	Trainrecall = 0.898	Devlogprob = -920.171	Devacc = 0.744	Devprec = 0.668	Devrecall = 0.939	
INFO:root:Update   3401	Trainlogprob = -3731.501	Trainacc = 0.737	Trainprec = 0.709	Trainrecall = 0.912	Devlogprob = -929.156	Devacc = 0.737	Devprec = 0.659	Devrecall = 0.948	
INFO:root:Update   3501	Trainlogprob = -3690.458	Trainacc = 0.743	Trainprec = 0.721	Trainrecall = 0.894	Devlogprob = -908.986	Devacc = 0.748	Devprec = 0.673	Devrecall = 0.937	
INFO:root:Update   3601	Trainlogprob = -3691.259	Trainacc = 0.743	Trainprec = 0.725	Trainrecall = 0.885	Devlogprob = -906.534	Devacc = 0.751	Devprec = 0.677	Devrecall = 0.932	
INFO:root:Update   3701	Trainlogprob = -3676.363	Trainacc = 0.745	Trainprec = 0.725	Trainrecall = 0.887	Devlogprob = -903.198	Devacc = 0.749	Devprec = 0.675	Devrecall = 0.932	
INFO:root:Update   3801	Trainlogprob = -3691.526	Trainacc = 0.743	Trainprec = 0.720	Trainrecall = 0.897	Devlogprob = -911.891	Devacc = 0.747	Devprec = 0.671	Devrecall = 0.938	
INFO:root:Update   3901	Trainlogprob = -3704.357	Trainacc = 0.744	Trainprec = 0.743	Trainrecall = 0.840	Devlogprob = -895.088	Devacc = 0.765	Devprec = 0.704	Devrecall = 0.890	
INFO:root:Update   4001	Trainlogprob = -3722.346	Trainacc = 0.732	Trainprec = 0.700	Trainrecall = 0.925	Devlogprob = -932.376	Devacc = 0.727	Devprec = 0.648	Devrecall = 0.956	
INFO:root:Update   4101	Trainlogprob = -3717.710	Trainacc = 0.735	Trainprec = 0.706	Trainrecall = 0.915	Devlogprob = -924.036	Devacc = 0.735	Devprec = 0.657	Devrecall = 0.947	
INFO:root:Update   4201	Trainlogprob = -3684.198	Trainacc = 0.744	Trainprec = 0.736	Trainrecall = 0.855	Devlogprob = -895.366	Devacc = 0.764	Devprec = 0.697	Devrecall = 0.908	
INFO:root:Update   4301	Trainlogprob = -3840.143	Trainacc = 0.714	Trainprec = 0.777	Trainrecall = 0.697	Devlogprob = -897.955	Devacc = 0.754	Devprec = 0.749	Devrecall = 0.742	
INFO:root:Update   4401	Trainlogprob = -3737.779	Trainacc = 0.745	Trainprec = 0.733	Trainrecall = 0.867	Devlogprob = -913.163	Devacc = 0.759	Devprec = 0.691	Devrecall = 0.913	
INFO:root:Update   4501	Trainlogprob = -3703.026	Trainacc = 0.741	Trainprec = 0.745	Trainrecall = 0.828	Devlogprob = -892.098	Devacc = 0.768	Devprec = 0.710	Devrecall = 0.882	
INFO:root:Update   4601	Trainlogprob = -3739.104	Trainacc = 0.746	Trainprec = 0.743	Trainrecall = 0.845	Devlogprob = -904.842	Devacc = 0.768	Devprec = 0.705	Devrecall = 0.897	
INFO:root:Update   4701	Trainlogprob = -3743.256	Trainacc = 0.744	Trainprec = 0.744	Trainrecall = 0.838	Devlogprob = -904.355	Devacc = 0.768	Devprec = 0.707	Devrecall = 0.890	
INFO:root:Update   4801	Trainlogprob = -3684.624	Trainacc = 0.742	Trainprec = 0.723	Trainrecall = 0.886	Devlogprob = -907.438	Devacc = 0.751	Devprec = 0.677	Devrecall = 0.930	
INFO:root:Update   4901	Trainlogprob = -3759.066	Trainacc = 0.727	Trainprec = 0.772	Trainrecall = 0.737	Devlogprob = -881.233	Devacc = 0.767	Devprec = 0.746	Devrecall = 0.786	
INFO:root:Update   5001	Trainlogprob = -3690.928	Trainacc = 0.743	Trainprec = 0.735	Trainrecall = 0.858	Devlogprob = -898.328	Devacc = 0.761	Devprec = 0.693	Devrecall = 0.911	
INFO:root:Update   5101	Trainlogprob = -3703.661	Trainacc = 0.731	Trainprec = 0.766	Trainrecall = 0.760	Devlogprob = -869.530	Devacc = 0.763	Devprec = 0.731	Devrecall = 0.810	
INFO:root:Update   5201	Trainlogprob = -3974.287	Trainacc = 0.682	Trainprec = 0.843	Trainrecall = 0.542	Devlogprob = -907.733	Devacc = 0.739	Devprec = 0.821	Devrecall = 0.591	
INFO:root:Update   5301	Trainlogprob = -3752.113	Trainacc = 0.740	Trainprec = 0.770	Trainrecall = 0.773	Devlogprob = -888.579	Devacc = 0.768	Devprec = 0.732	Devrecall = 0.824	
INFO:root:Update   5401	Trainlogprob = -3712.258	Trainacc = 0.745	Trainprec = 0.722	Trainrecall = 0.897	Devlogprob = -913.111	Devacc = 0.744	Devprec = 0.671	Devrecall = 0.930	
INFO:root:Update   5501	Trainlogprob = -3710.939	Trainacc = 0.748	Trainprec = 0.733	Trainrecall = 0.876	Devlogprob = -904.010	Devacc = 0.750	Devprec = 0.681	Devrecall = 0.913	
INFO:root:Update   5601	Trainlogprob = -3736.990	Trainacc = 0.728	Trainprec = 0.692	Trainrecall = 0.939	Devlogprob = -941.500	Devacc = 0.718	Devprec = 0.639	Devrecall = 0.961	
INFO:root:Update   5701	Trainlogprob = -3680.221	Trainacc = 0.742	Trainprec = 0.716	Trainrecall = 0.906	Devlogprob = -911.949	Devacc = 0.743	Devprec = 0.666	Devrecall = 0.943	
INFO:root:Update   5801	Trainlogprob = -3686.681	Trainacc = 0.740	Trainprec = 0.714	Trainrecall = 0.907	Devlogprob = -910.948	Devacc = 0.740	Devprec = 0.664	Devrecall = 0.942	
INFO:root:Update   5901	Trainlogprob = -3665.866	Trainacc = 0.745	Trainprec = 0.722	Trainrecall = 0.896	Devlogprob = -900.597	Devacc = 0.741	Devprec = 0.668	Devrecall = 0.930	
INFO:root:Update   6001	Trainlogprob = -3704.820	Trainacc = 0.742	Trainprec = 0.714	Trainrecall = 0.910	Devlogprob = -913.823	Devacc = 0.738	Devprec = 0.662	Devrecall = 0.941	
INFO:root:Update   6101	Trainlogprob = -3860.515	Trainacc = 0.727	Trainprec = 0.812	Trainrecall = 0.677	Devlogprob = -900.602	Devacc = 0.778	Devprec = 0.788	Devrecall = 0.742	
INFO:root:Update   6201	Trainlogprob = -3676.337	Trainacc = 0.744	Trainprec = 0.719	Trainrecall = 0.902	Devlogprob = -906.039	Devacc = 0.743	Devprec = 0.667	Devrecall = 0.939	
INFO:root:Update   6301	Trainlogprob = -3681.140	Trainacc = 0.746	Trainprec = 0.726	Trainrecall = 0.890	Devlogprob = -902.222	Devacc = 0.748	Devprec = 0.674	Devrecall = 0.930	
INFO:root:Update   6401	Trainlogprob = -3820.547	Trainacc = 0.718	Trainprec = 0.679	Trainrecall = 0.958	Devlogprob = -979.029	Devacc = 0.697	Devprec = 0.620	Devrecall = 0.973	
INFO:root:Update   6501	Trainlogprob = -3723.399	Trainacc = 0.731	Trainprec = 0.696	Trainrecall = 0.934	Devlogprob = -936.832	Devacc = 0.723	Devprec = 0.644	Devrecall = 0.961	
INFO:root:Update   6601	Trainlogprob = -3752.384	Trainacc = 0.725	Trainprec = 0.687	Trainrecall = 0.946	Devlogprob = -953.565	Devacc = 0.708	Devprec = 0.630	Devrecall = 0.967	
INFO:root:Feat         Category_category:Fine Arts   1: -0.02808
INFO:root:Feat          Category_category:Religion   7: -0.00339
INFO:root:Feat                        Length_guess  14: -0.06532
INFO:root:Feat    Category_category:Social Science   9: -0.00332
INFO:root:Feat        Category_category:Literature   4: -0.03398
INFO:root:Feat         Category_category:Mythology   5: -0.02300
INFO:root:Feat           Category_category:Science   8: -0.02205
INFO:root:Feat                       BIAS_CONSTANT   0: -0.00153
INFO:root:Feat           Category_category:History   3: +0.01173
INFO:root:Feat        Category_category:Philosophy   6: +0.01742
INFO:root:Feat           Category_category:Science   8: -0.02205
INFO:root:Feat                       BIAS_CONSTANT   0: -0.00153
INFO:root:Feat           Category_category:History   3: +0.01173
INFO:root:Feat        Category_category:Philosophy   6: +0.01742
INFO:root:Feat         Category_category:Geography   2: +0.05906
INFO:root:Feat                          Length_ftp  13: +0.15851
INFO:root:Feat                      Gpr_confidence  11: +0.19554
INFO:root:Feat                         Length_char  12: +0.26772
INFO:root:Feat                     Frequency_guess  10: +0.52527
INFO:root:Feat                         Length_word  15: +0.26101
```

Hints
-

1.  As with the previous assignment, make sure that you debug on small
    datasets first (I've provided _toy text_ in the data directory to get you started).
1.  Use numpy functions whenever you can to make the computation
faster.




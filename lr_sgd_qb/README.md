Stochastic Gradient Descent
=

Overview
--------

In this homework you'll implement a stochastic gradient ascent for
logistic regression and you'll apply it to the task of determining
whether an answer to a question is correct or not.

This will be slightly more difficult than the last homework (the difficulty
will slowly ramp upward).  You should not use any libraries that implement any
of the functionality of logistic regression for this assignment; logistic
regression is implemented in scikit learn, pytorch, and many other places, but
you should do everything by hand now.  You'll be able to use library
implementations of logistic regression in the future.

You'll turn in your code on Gradescope.  This assignment is worth 30 points.

What you have to do
----

Coding (25 points):

1. Understand how the code is creating feature vectors (this will help you
code the solution and to do the later analysis).  You don't actually need to
write any code for this, however.

2. (Optional) Store necessary data in the constructor so you can do
classification later.

3. You'll likely need to write some code to get the best/worst features (see
below).

3. Modify the _sg update_ function to perform updates.

Analysis (5 points):

1. What is the role of the learning rate?
2. How many datapoints (or multiple passes over the data) do you need to
complete for the *model* to stabilize?  The various metrics can give you clues
about what the model is doing, but no one metric is perfect.
3. What features are the best predictors of each class?  How (mathematically)
did you find them?
4. What features are the poorest predictors of classes?  How (mathematically)
did you find them?

Extra credit:

1.  Modify the _sg update_ function to perform [lazy regularized updates](https://lingpipe.files.wordpress.com/2008/04/lazysgdregression.pdf), which only update the weights of features when they appear in an example.
    - Show the effect in your analysis document 
    
Caution: When implementing extra credit, make sure your implementation of the
regular algorithms doesn't change.

What to turn in
-

1. Submit your _lr_sgd_qb.py_ file (include your name at the top of the source)
1. Submit your _analysis.pdf_ file
    - no more than one page (NB: This is also for the extra credit.  To minimize effort for the grader, you'll need to put everything on a page.  Take this into account when selecting if/which extra credit to do...think of the page requirement like a regularizer).
    - pictures are better than text
    - include your name at the top of the PDF

Unit Tests
=

I've provided unit tests based on the example that we worked through
in class.  Before running your code on read data, make sure it passes
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
Loaded 813 items from vocab ../data/small_guess.vocab
Read in 82395 train and 4820 test
Update 1	TProb -58226.627871	HProb -3329.405842	TAcc 0.340979	HAcc 0.556846
Update 101	TProb -51280.344230	HProb -3587.838259	TAcc 0.659640	HAcc 0.443154
Update 201	TProb -50900.479422	HProb -3491.855595	TAcc 0.666727	HAcc 0.453112
Update 301	TProb -50675.157120	HProb -3573.737948	TAcc 0.664009	HAcc 0.447925
Update 401	TProb -49747.982642	HProb -3588.104068	TAcc 0.661254	HAcc 0.446473
Update 501	TProb -49718.270414	HProb -3739.966407	TAcc 0.662103	HAcc 0.448133
Update 601	TProb -48885.690172	HProb -3725.017894	TAcc 0.670344	HAcc 0.449378
Update 701	TProb -50877.398237	HProb -4103.115825	TAcc 0.660064	HAcc 0.446888
Update 801	TProb -48137.228124	HProb -3460.385782	TAcc 0.680709	HAcc 0.514315
Update 901	TProb -49239.832616	HProb -3879.116805	TAcc 0.665841	HAcc 0.448963
Update 1001	TProb -47564.284603	HProb -3454.699256	TAcc 0.688112	HAcc 0.523859
Update 1101	TProb -48278.387339	HProb -3220.855567	TAcc 0.713830	HAcc 0.600622
Update 1201	TProb -47310.437949	HProb -3258.288936	TAcc 0.714558	HAcc 0.602490
Update 1301	TProb -46384.112620	HProb -3222.882106	TAcc 0.710674	HAcc 0.617635
Update 1401	TProb -45959.382377	HProb -3118.785124	TAcc 0.711633	HAcc 0.642531
Update 1501	TProb -47567.666705	HProb -2950.560708	TAcc 0.729377	HAcc 0.683817
Update 1601	TProb -47406.282279	HProb -2955.135330	TAcc 0.716051	HAcc 0.685685
Update 1701	TProb -46627.653538	HProb -2954.608999	TAcc 0.730603	HAcc 0.682158
Update 1801	TProb -45890.055129	HProb -2984.371207	TAcc 0.722932	HAcc 0.672822
Update 1901	TProb -46140.877944	HProb -3044.769170	TAcc 0.697008	HAcc 0.649378
Update 2001	TProb -46293.884993	HProb -2965.970627	TAcc 0.710941	HAcc 0.675934
Update 2101	TProb -46735.076092	HProb -2944.407308	TAcc 0.721087	HAcc 0.679876
Update 2201	TProb -46191.998867	HProb -2932.935651	TAcc 0.733078	HAcc 0.686515
Update 2301	TProb -45059.271994	HProb -3005.847730	TAcc 0.723284	HAcc 0.671577
Update 2401	TProb -45148.730886	HProb -2952.490624	TAcc 0.738334	HAcc 0.682573
Update 2501	TProb -45858.990372	HProb -3092.898373	TAcc 0.699108	HAcc 0.656224
Update 2601	TProb -44530.771708	HProb -2905.384097	TAcc 0.737059	HAcc 0.691286
Update 2701	TProb -44566.167455	HProb -2905.772393	TAcc 0.733892	HAcc 0.691079```

Hints
-

1.  As with the previous assignment, make sure that you debug on small
    datasets first (I've provided _toy text_ in the data directory to get you started).
1.  Use numpy functions whenever you can to make the computation
faster.




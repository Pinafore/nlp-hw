K-Nearest Neighbors
=

Overview
--------

In this homework you'll implement a k-Nearest Neighbor classification
framework to take a question, find the most similar question to it,
and answer with the answer to that question.

This is designed to be a *very easy* homework.  If you're spending a
lot of time on this assignment, you are either:

* not prepared to take the course 
* seriously over-thinking the assignment
* trying to reimplement too much of the assignment

Most of this assignment will be done by calling libraries that have
already been implemented for you.  If you are implementing
n-dimensional search or a median algorithm, you are generating extra
work for yourself and making yourself vulnerable to errors.

You'll turn in your code on the submit server.

What you have to do
----

Coding (20 points):

1.  (Optional) Store necessary data in the constructor so you can do classification later.
1.  Modify the _majority_ function so that it returns the *value* associated with the most *indices*.  If there's a tie, return the one that's alphabetically first.
1.  Modify the _classify_ function so that it finds the closest indicies (in terms of *cosine* similarity) to the query point.
1.  Modify the _confusion matrix_ function to classify examples and record which number it got right.

Analysis (5 points):

1.  What is the role of the number of training points to accuracy?
1.  What is the role of _k_ to accuracy?
1.  What answers get confused with each other most easily?

What you don't have to do
-------
You don't have to (and shouldn't!) compute tf-idf yourself.  We did that in the last homework, so you can leave that to the professionals.  We provide code that does this in the main function.  You probably shouldn't modify it, but it's probably useful to understand it for future homeworks (you'll need to write/call code like it in the future).

What to turn in
-

1.  Submit your _knn.py_ file
1.  Submit your _analysis.pdf_ file (no more than one page; pictures
    are better than text)

Extra Credit
=
You can get extra credit for submitting your system on CodaLab.  Submit a separate version of your code that you submitted on CodaLab.

http://leaderboard.qanta.org

You're free to modify the code however you would like.  You can get up to 10 points of extra credit depending on how well you do.  E.g., if you just submit the code as is, you'll get three points.  Doing something extra (e.g., tuning parameters) will get you more points.

Unit Tests
=

I've provided unit tests based on the example that we worked through
in class.  Before running your code on read data, make sure it passes
all of the unit tests.


```
cs244-33-dhcp:knn jbg$ python tests.py
...
----------------------------------------------------------------------
Ran 3 tests in 0.003s

OK
```

Initially, it will fail all of them:
```
cs244-33-dhcp:knn jbg$ python tests.py
FFF
======================================================================
FAIL: test1 (__main__.TestKnn)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests.py", line 20, in test1
    self.assertEqual(self.knn[1].classify(self.queries[1]), -1)
AssertionError: 1 != -1

======================================================================
FAIL: test2 (__main__.TestKnn)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests.py", line 25, in test2
    self.assertEqual(self.knn[2].classify(self.queries[0]), 1)
AssertionError: -1 != 1

======================================================================
FAIL: test3 (__main__.TestKnn)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests.py", line 31, in test3
    self.assertEqual(self.knn[3].classify(self.queries[0]), 1)
AssertionError: -1 != 1

----------------------------------------------------------------------
Ran 3 tests in 0.002s

FAILED (failures=3)
```

Example
-

This is an example of what your code (knn.py) output should look like:
```
Pranavs-MacBook-Pro:knn2 pranavgoel$ python knn.py 
<class 'scipy.sparse.csr.csr_matrix'>
Done loading data
100/100 for confusion matrix
	Texas_annexation	Mark_Antony	Martin_Scorsese	Spin_(physics)	Operation_Condor
------------------------------------------------------------------------------------------
                   Mark_Antony:	0	1	0	0	0
                     Dirty_War:	0	0	0	0	1
              Texas_annexation:	1	0	0	0	0
                Spin_(physics):	0	0	0	1	0
               Martin_Scorsese:	0	0	1	0	0
Accuracy: 0.480000
```

Hints
-

1.  Don't use all of the data, especially at first.  Use the _limit_
    command line argument (as in the above example).  We'll be using
    this dataset again with techniques that scale better.
1.  Don't reimplement closest point data structures or median.
1.  Make sure your code actually behaves differently for different
    values of _k_.
1.  You can use dot product only if the vectors start off normalized. 

Nearest Neighbor Question Answering (Guessing)
=

Overview
--------

In this homework you'll implement a nearest neighbor approach to answering
questions.  Given a question you find the most similar question to it, and
answer with the answer to that question.

This is designed to be a *very easy* homework.  If you're spending a
lot of time on this assignment, you are either:

* not prepared to take the course 
* seriously over-thinking the assignment
* trying to reimplement too much of the assignment

Most of this assignment will be done by calling libraries that have already
been implemented for you.  If you are over-implementing, you are generating
extra work for yourself and making yourself vulnerable to errors.

You'll turn in your code on Gradescope.

What you have to do
----

Coding (20 points):

1.  (Optional) Store necessary data in the constructor so you can do classification later.
1.  Modify the _train_ function so that the class stores what it needs to store to guess at what the answer is.
1.  Modify the _guess_ function so that it finds the closest indicies (in terms of *cosine* similarity) to the query point.

Analysis (5 points):

1.  What is the role of the number of training points to accuracy?
1.  What answers get confused with each other most easily?
1.  Compute precision and recall as you increase the number of guesses.

What you don't have to do
-------

You don't have to (and shouldn't!) compute tf-idf yourself.  We did that in
the last homework, so you can leave that to the professionals.  We encourage
you to use the tf-idf vectorizer: play around with different settings of the paramters.  You probably shouldn't modify it,
but it's probably useful to understand it for future homeworks (you'll need to
write/call code like it in the future).

What to turn in
-

1.  Submit your _tfidf_guesser.py_ file
1.  Submit your _analysis.pdf_ file (no more than one page; pictures
    are better than text)

Extra Credit
=
You can get extra credit for submitting your system on the Codalab leaderboard.

You're free to modify the code however you would like.  You can get up to 10
points of extra credit depending on how well you do.  E.g., if you just submit
the code as is, you'll get three points.  Doing something extra (e.g., tuning
parameters) will get you more points.

Unit Tests
=

I've provided unit tests.  Before running your code on real data, make sure it
passes all of the unit tests.


```
python3 tfidf_guesser_test.py 
..
----------------------------------------------------------------------
Ran 2 tests in 0.302s

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
jbg@wuzetian:~/repositories/courses/cmsc_470/solutions> python3 tfidf_guesser.py 
Loading data/small.guesstrain.json
100/227 for confusion matrix
200/227 for confusion matrix
Errors:
=================================================
1	Netherlands	Japan	
1	United_States	Cuba	
1	Japan	Egypt	
1	Death	Emily_Dickinson	
1	Death	Water	
1	Indonesia	Argentina	
1	Indonesia	Cuba	
1	Hamlet	T._S._Eliot	
1	Frank_Lloyd_Wright	Frank_Gehry	
1	Francisco_Goya	Pieter_Bruegel_the_Elder	
2	Dog	Horse	
1	Germany	Spain	
1	Superconductivity	Semiconductor	
1	Myanmar	England	
1	Italy	Japan	
1	Poland	Italy	
1	San_Francisco	New_York_City	
2	China	Spain	
1	China	Japan	
1	London	Iran	
1	Joseph_Haydn	Franz_Schubert	
1	Victor_Hugo	Willa_Cather	
1	Belgium	Japan	
1	Vienna	Prague	
1	Vienna	Los_Angeles	
1	Gustav_Mahler	Franz_Schubert	
1	Auxin	Radical_(chemistry)	
1	Spain	India	
1	Spain	Egypt	
1	Wolfgang_Amadeus_Mozart	Ludwig_van_Beethoven	
1	Triangle	Black_hole	
1	Aldehyde	Alkene	
1	4	3	
1	Athens	Sparta	
1	Surface_tension	Mass	
1	Aztec	Maya_civilization	
1	Pressure	Temperature	
1	Daniel_Defoe	Isabel_Allende	
1	Sikhism	Jainism	
1	Thebes,_Greece	Prague	
1	One_Thousand_and_One_Nights	Russia	
1	Cambodia	Thailand	
1	Nigeria	Wole_Soyinka	
1	Carthage	Sparta	
1	Egypt	Greece	
1	Copper	Iron	
1	Plasma_(physics)	Colloid	
```

Hints
-

1.  Don't use all of the data, especially at first.  Use the _limit_
    command line argument (as in the above example).  We'll be using
    this dataset again with techniques that scale better.
1.  You probably want to tune tf-idf parameters.  Play around with what works well!
1.  You can use dot product only if the vectors start off normalized. 

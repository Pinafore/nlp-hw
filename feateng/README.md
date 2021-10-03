Feature Engineering
=

The goal of this assignment is to take the pieces of code that we had from
previous homeworks, but them together, and make them better.

You will build on the *tf-idf guesser* by extracting useful information from
its guesses, generate better features for input into the *pytorch logistic
regression* classifier to do a better job of selecting whether a guess to a
question is correct.

NOTE: Because the goal of this assignment is feature engineering, not classification algorithms, you may not change the underlying algorithm.

It is structured in a way that approximates how classification works in the real world: Features are typically underspecified (or not specified at all) You, the data digger, have to articulate the features you need You then compete against others to provide useful predictions

It may seem straightforward, but do not start this at the last minute. There are often many things that go wrong in testing out features, and you'll want to make sure your features work well once you've found them.

What Can You Do?
-

You can:
* Add features
* Change feature representations
* Exclude training data
* Add training data

What Can't You Do?
-
Remove tf-idf as guesser or logistic regression as buzzer.

Accuracy (15+ points)
------------------------------

15 points of your score will be generated from your performance on the
the classification competition on the leaderboard.  The performance will be
evaluated on accuracy on a held-out test set.

You should be able to significantly
improve on the baseline system.  If you can
do much better than your peers, you can earn extra credit (up to 10 points).

Analysis (10 Points)
--------------

The job of the written portion of the homework is to convince the grader that:
* Your new features work
* You understand what the new features are doing
* You had a clear methodology for incorporating the new features

Make sure that you have examples and quantitative evidence that your
features are working well.  Be sure to explain how used the data
(e.g., did you have a development set) and how you inspected the
results.

A sure way of getting a low grade is simply listing what you tried and
reporting the Kaggle score for each.  You are expected to pay more
attention to what is going on with the data and take a data-driven
approach to feature engineering.

How to Turn in Your System
-
Turn in your system as usual via Gradescope, where we'll be using the leaderboard as before.  However, this time the score from the leaderboard will be part of your grade.

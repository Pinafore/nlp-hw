Feature Engineering
=

The goal of this assignment is to take the QA system components (i.e guesser, buzzer) we had from previous homeworks, put them together, and make them better. 

You will build on the *tf-idf guesser* by extracting useful information from
its guesses and generate better features for input into the *pytorch logistic
regression* classifier to do a better job of selecting whether a guess to a
question is correct.

NOTE: Because the goal of this assignment is feature engineering, not classification algorithms, you may not change the underlying algorithm.

This assignment is structured in a way that approximates how classification works in the real world: Features are typically underspecified (or not specified at all). You, the data digger, have to articulate the features you need. You then compete against others to provide useful predictions.

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
features are working well, and include the metrics you chose to measure your system's performance. Be sure to explain how used the data
(e.g., did you have a development set?) and how you inspected the
results.

A sure way of getting a low grade is simply listing what you tried and
reporting the Kaggle score for each.  You are expected to pay more
attention to what is going on with the data and take a data-driven
approach to feature engineering.

How to Turn in Your System
-
If you used additional training data, please include the source data in your Gradescope submission in a file named ``test_custom.json`` (if your file is <100MB), or submit a shell script named ``gather_resources.sh`` that will retrieve the file ``test_custom.json`` programatically from a public location (i.e a public S3 bucket).

In addition, include your trained model in your submission by submitting a file named ``trained_model.th``. Follow the instructions [here](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference) on how to do this (saving and loading the ``state_dict``). **If you do not correctly save your trained model, or do not submit one at all, the autograder will fail.**

Turn in your system as usual via Gradescope, where we'll be using the leaderboard as before.  However, this time the score from the leaderboard will be part of your grade.


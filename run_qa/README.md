
Homework: Run somebody else's QA system
========================================

The goal of this homework is to run a "real" QA system on our crazy
quiz bowl data.

Is it okay if this overlaps with my project
===========================================

Not only is it okay, it's highly recommended.  However, everybody
needs to do their own submission (as usual).  You can talk to each
other, but the code must come from your fingertips.

Since many of your projects want to try out different models,
different team members can try out slightly different models and see
what it takes to submit them on CodaLab.

How do I choose a Model?
========================

There are a couple of concerns that you need to balance: how good is
the model, how well will it work on our funky QB data, and how
complicated will it be to get it to run at all.  Don't bite off more
than you can chew.  Select a model that isn't too complicated, select
one that's flexible, and don't care too much about reported accuracy.

Also ignore sunk costs, if it doesn't look like this model will work,
feel free to try another one.

There are also size constraints on CodaLab submissions; make sure your
model fits!

What to Turn In
===============

* Submit your system to CodaLab
* Write a one page document describing what you did to get it to run,
  what changes you made to make it work better for quiz bowl questions

Grading
=======

The denominator is 15 points.  However, lots of extra credit is
available if you do well (place well on the leaderboard): up to 20
additional points.

For the required points, all you need to do is: get a system running,
submit it Codalab, and get it work on our crazy quiz bowl data.  It
doesn't matter how well it works as long as it gets non-negligible
accuracy.

The extra credit is available for doing well in terms of the expected
wins metric.

Hints
====

Here are some systems you could try:
* https://github.com/allenai/macaw
* https://github.com/ad-freiburg/aqqu
* https://github.com/castorini/pyserini/blob/master/docs/experiments-ance.md
* https://docs.allennlp.org/models/main/models/rc/predictors/bidaf/
* https://github.com/fastforwardlabs/ff14_blog/blob/master/_notebooks/2020-05-19-Getting_Started_with_QA.ipynb


You should carefully look at this tf-idf baseline:
https://github.com/Pinafore/qanta-codalab
(note that you can't just submit this system, you need to submit
something other than tf-idf.)

FAQ
===

* I did this for the tf-idf guesser extra credit, can I just count that?

No, but it should be much easier for you!  It must be a model that somebody else wrote.

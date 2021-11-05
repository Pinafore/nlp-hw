
Homework: Run someone else's QA system
========================================

The goal of this homework is to get a "real" QA system running on our crazy
quiz bowl data. You'll submit the model you choose to our qanta leaderboard via CodaLab.

Is it okay if this overlaps with our course project?
====================================================

Not only is it okay, it's highly recommended.  However, as usual, everybody must submit their own 
system. You can talk to each other, but the code must come from your own fingertips.

Since many groups want to try out a few different models,
different team members can experiment with different models and see
what it takes to submit them on CodaLab.

How do I choose a QA model?
===========================

There are a few concerns you'll need to balance: how strong the model is, 
how well it will work on our funky Quizbowl data, and how
complicated or involved it'll be to get it to run at all.  Don't bite off more
than you can chew.  Select a model that isn't too complicated, select
one that's flexible, and don't worry too much about reported accuracy.

Also ignore sunk costs – if it doesn't look like this model will work,
feel free to try another one.

There are also size constraints on CodaLab submissions; make sure your
model fits!

What to Turn In
===============

* System submission: Submit your system to CodaLab. See [here](https://github.com/Pinafore/qanta-codalab#codalab) for submission instructions.
* `writeup.pdf`: Write a one page document describing the system you chose, what you did to get it to run, and 
  the changes you made to make it work better for quiz bowl questions. **Please include the name of your leaderboard submission (i.e the name you choose for the bundle description) at the top of your writeup. If you do not do this, we will not be able to link your submission to your writeup.**

Grading
=======

The assignment is out of 15 points. However, up to 20 additional points of extra credit are available if you
place well on the leaderboard.

To earn the 15 required points, all you need to do is: 
* get a system running
* submit it Codalab
* get it to work on our crazy quiz bowl data

It doesn't matter how _well_ it works as long as it achieves non-negligible
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

* I did this for the tf-idf guesser extra credit, can I just submit that?

No, but it should be much easier for you!  It must be a model that somebody else wrote.

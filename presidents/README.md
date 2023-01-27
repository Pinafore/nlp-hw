
Who was President When?
=======================

To get us warmed up, we're going to practice using a knowledge base (the dates
of US presidents in office) and string manipulation.  We'll also get used to
Gradescope and its leaderboard functionality.  More importantly, this code
will introduce some of the object-oriented structure of future projects.

This homework is meant to be very easy.  You should not need to spend a lot of
time on this.

However, like many of our homework assignments, there will be opportunity for
extra credit.

What you have to do
===================

The PresidentGusser takes a question and returns a guess of who the president
was.  You need to extract what time the question is asking about and return
the name of the appropriate US president.

This should be very simple, no more than five lines of code.  If you're
writing far more than that, you're likely not taking advantage of built-in
Python libraries that you should be using.

How do I know if my code is working?
====================================

Run `python test.py` on the command line and fix any tests are failing.  While
this is a neccessary condition to getting full credit on the assignment, it is
not sufficient.

How to turn it in
=================

Modify `president_guesser.py` and upload it HW0 on Gradescope.


Frequently Asked Questions
==========================

**Q: What time formats do we need to deal with?**

**A**: All of the all of the questions will be formatted as `Day Mon DD
  HH:MM:SS YYYY`.  Look at the unit tests for examples.

**Q: Can I change the training data or the logic in loading the training
  data?**

**A**: You may, but you shouldn't need to do this for anything but the extra
  credit.

**Q: I've done the homework, passing the tests, but I'm not getting a perfect
  score on the leaderboard.  How do I do better to get extra credit?**

**A**: The whole point of extra credit is to be fun / challenging.  Consider
  assumptions and limitations of the template code.

**Q: When does a president's term start?**

**A:** The twentieth amendment states that the term of each elected President
  of the United States begins at noon on January 20.  Before that, we assume
  it started at noon on the day they were innaugurated.

**Q: Okay, I'm doing better, but I'm still not perfect.**

**A:**: Some of the tests are indeed trickier.

**Q: Is there some president that's not in the provided training data?**

**A:**: All of the possible presidents are listed there.  

**Q: In 1849, who was president the evening of March 4 1849 and the morning of
  March 5?**

**A:** It is tempting to say that [David Rice
  Atchison](https://en.wikipedia.org/wiki/David_Rice_Atchison), but most
  constitutional scholars believe that Zachary Taylor was nonetheless
  president despite not taking the oath on Sunday.

Points Possible
===============

You get full credit for matching the baseline accuracy (85%) and can get up to
three points for improving significantly beyond that.

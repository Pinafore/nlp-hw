
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

There are two things you need to do: store the appropriate information
in the `train` function and then retrieve it in the `__call__` function.

This should be very simple, no more than five lines of code.  If you're
writing far more than that, you're likely not taking advantage of built-in
Python libraries that you should be using.

What's "Good Enough"?
====================================

This homework will also introduce you to the "Good Enough" goals of an
assignment.  If you do this, you'll get an A.  You won't necessarily get all
of the possible points and certainly not the extra credit, but if you do this
much, you can certainly stop stressing about the assignment.  

For each homework, we'll outline what the "Good Enough" goals are;
this is important because most of the homeworks are fairly open
ended.  It's important for you to know when it's okay to call it a
day.

For this homework, just list the correct presidents given the year and
return all possible presidents in a list.  We've made it easy for you
do do that:

    def __call__(self, question, n_guesses=1):
        # Update this code so that we can have a different president than Joe
        # Biden
        candidates = ["Joseph R. Biden"]

        if len(candidates) == 0:
            return [{"guess": ""}]
        else:
            return [{"guess": x} for x in candidates]

Just fill in all of the possible presidents given that year in the
`candidates` list and return them.  For many of our homeworks, getting
the right answer somewhere in the list of top results is more
important than just getting ~the one~ answer.

How do I know if my code is working?
====================================

For most of your homeworks, there will be two ways to test your code:
unit tests and `eval.py`.  For this homework, the feedback is exactly
the same, but for future homeworks, they will provide different
information.

Let's start with the unit tests first.  When you run them, you should
see something like this:

     jbg@MacBook-Pro-von-Jordan presidents % python3 president_test.py
     F
     ======================================================================
     FAIL: test_basic (__main__.TestPresidentGuessers.test_basic)
     ----------------------------------------------------------------------
     Traceback (most recent call last):
       File "/Users/jbg/repositories/nlp-hw/presidents/president_test.py", line 13, in test_basic
         self.assertEqual(guess, ii["page"])
     AssertionError: 'Joseph R. Biden' != 'Ronald Reagan'
     - Joseph R. Biden
     + Ronald Reagan
      : Wrong answer for: Who was president on Sat May 23 02:00:00 1982?

     ----------------------------------------------------------------------
     Ran 1 test in 0.000s

The starter code will always return that Joe Biden is the president,
and it got the answer wrong for May 23, 1982.  As you fix those
issues, it will only show the first test that you fail, so eventually
it will say that all tests pass.

But unit tests aren't really the best tests, so you should also run
the eval script.  If you've met the "good enough" goals, you should
get a result that looks like this:

    jbg@MacBook-Pro-von-Jordan GPT3QA % python3 eval.py --evaluate=guesser --guesser_type='PresidentGuesser' --questions=presidents --question_source='toy'
    Setting up logging
    INFO:root:Read 9 questions
    INFO:root:Using device 'cpu' (cuda flag=False)
    INFO:root:Initializing guesser of type PresidentGuesser
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 80487.71it/s]
    INFO:root:Generating guesses for 9 new question
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 16049.63it/s]
    close 1.00
    ===================
    
                   guess: George W. Bush
                  answer: Barack Obama
                      id: 204
                    text: Who was president on Tue Jan 20 13:00:00 2009?
    --------------------
                   guess: John F. Kennedy
                  answer: Lyndon B. Johnson
                      id: 205
                    text: Who was president on Fri Nov 22 16:00:00 1963?
    --------------------
                   guess: Herbert Hoover
                  answer: Franklin D. Roosevelt
                      id: 207
                    text: Who was president on Sat Mar 04 21:00:00 1933?
    --------------------
                   guess: Abraham Lincoln
                  answer: Andrew Johnson
                      id: 208
                    text: Who was president on Sat Apr 15 15:00:00 1865?
    --------------------
    =================
    hit 0.56
    ===================
    
                   guess: Joseph R. Biden
                  answer: Joseph R. Biden
                      id: 201
                    text: Who was president on Wed Jan 25 06:20:00 2023?
    --------------------
                   guess: Ronald Reagan
                  answer: Ronald Reagan
                      id: 202
                    text: Who was president on Sat May 23 02:00:00 1982?
    --------------------
                   guess: Joseph R. Biden
                  answer: Joseph R. Biden
                      id: 203
                    text: Who was president on Wed Mar 01 04:23:40 2023?
    --------------------
                   guess: Harry S. Truman
                  answer: Harry S. Truman
                      id: 206
                    text: Who was president on Tue Apr 12 20:00:00 1949?
    --------------------
                   guess: George Washington
                  answer: George Washington
                      id: 209
                    text: Who was president on Thu Apr 30 17:00:00 1789?
    --------------------
    =================
    Precision @1: 0.5556 Recall: 1.0000
    
We'll talk about exactly what precision and recall mean in future
lectures, but Precision @1 corresponds to how often the right answer
was at the top of your list of results and recall is how often the
right answer appeared in that list.  

It should be possible to get 1.0 Precision@1 for this homework (but
nott in future homeworks).  But again, this is not required for a
"Good Enough" submission.

How to turn it in
=================

Modify `president_guesser.py` and upload it HW0 on Gradescope.  Do not
modify any other files.

Frequently Asked Questions
==========================

**Q: I'm getting an error message saying that `foobar` cannot be
imported.  What do I do?**

**A**: Create a virtual environment to (install the
packages)[https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/].

    python3 -m pip install -r requirements.txt

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

**A:** It is tempting to say [David Rice
  Atchison](https://en.wikipedia.org/wiki/David_Rice_Atchison), but most
  constitutional scholars believe that Zachary Taylor was nonetheless
  president despite not taking the oath on Sunday.

Points Possible
===============

You get full credit for matching the baseline accuracy (85%) and can get up to
three points for improving significantly beyond that.

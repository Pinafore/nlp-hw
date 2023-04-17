The evaluation of this project is to answer trivia questions.  You do
not need to do well at this task, but you should submit a system that
completes the task or create adversarial questions in that setting.  This will help the whole class share data and
resources.

If you focus on something other than predicting answers, *that's fine*!  

About the Data
==============

Quiz bowl is an academic competition between schools in
English-speaking countries; hundreds of teams compete in dozens of
tournaments each year. Quiz bowl is different from Jeopardy, a recent
application area.  While Jeopardy also uses signaling devices, these
are only usable after a question is completed (interrupting Jeopardy's
questions would make for bad television).  Thus, Jeopardy is rapacious
classification followed by a race---among those who know the
answer---to punch a button first.

Here's an example of a quiz bowl question:

Expanding on a 1908 paper by Smoluchowski, he derived a formula for
the intensity of scattered light in media fluctuating densities that
reduces to Rayleigh's law for ideal gases in The Theory of the
Opalescence of Homogenous Fluids and Liquid Mixtures near the Critical
State.  That research supported his theories of matter first developed
when he calculated the diffusion constant in terms of fundamental
parameters of the particles of a gas undergoing Brownian Motion.  In
that same year, 1905, he also published On a Heuristic Point of View
Concerning the Production and Transformation of Light.  That
explication of the photoelectric effect won him 1921 Nobel in Physics.
For ten points, name this German physicist best known for his theory
of Relativity.

*ANSWER*: Albert _Einstein_

Two teams listen to the same question. Teams interrupt the question at
any point by "buzzing in"; if the answer is correct, the team gets
points and the next question is read.  Otherwise, the team loses
points and the other team can answer.

You are welcome to use any *automatic* method to choose an answer.  It
need not be similar nor build on our provided systems.  In addition to
the data we provide, you are welcome to use any external data *except*
our test quiz bowl questions (i.e., don't hack our server!).  You are
welcome (an encouraged) to use any publicly available software, but
you may want to check on Piazza for suggestions as many tools are
better (or easier to use) than others.

If you don't like the interruptability of questions, you can also just answer entire questions.  However, you must also output a confidence.

Competition
==================
We will use Dynabech website (https://dynabench.org/tasks/qa). If you remember the past workshop about Dynabench submission, this is the way to do it. The specific task name is "Grounded QA". Here, with the help of the video tutorial, you submit your QA model and assess how your QA model did compared to others. The assessment will take place by testing your QA model on several QA test datasets and the results of yours and your competitors will be visible on the leaderboard. Your goal is to rank the highest in terms of F1 score and accuracy. 

Here is the Dynabench tutorial. https://umd.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=91792791-de2e-48ee-a210-afb900212d5d
(As of now, the task is not yet published, but you can always get a head start with the existing QA task to test out your model or practice the submission preocess before the deadline (https://dynabench.org/tasks/qa). 

Writing Questions
==================

Alternatively, you can also *write* 50 adversarial questions that challenge modern NLP systems.  These questions must be diverse in the subjects asked about, the skills computers need to answer the questions, and the entities in those questions.

In addition to the raw questions, you will also need to create citations describing:
* Why the question is difficult for computers
* Why the information in the question is correct
* Why the question is interesting
* Why the question is pyramidal
* And a non-pyramidal question that could be used for something like OQL (OQL contains topics such as: Art and Literature, Geography, History, Science, TV and Film, Music, Lifestyle, and Sport. )
* Run state of the art QA systems on the questions to show they struggle, give individual results for each question and a summary over all questions

For an example of what the writeup should look like, see the adversarial HW:
https://github.com/Pinafore/nlp-hw/blob/master/adversarial/question.tex

Proposal
==================

The project proposal is a one page PDF document that describes:

* Who is on your team

* What techniques you will explore 

* Your timeline for completing the project (be realistic; you should
  have your first submission in a week or two)

Have the person whose last name is alphabetically last submit the
proposal on Piazza.  Late days cannot be used on this
assignment.

First Deliverable
====================== 

You'll have to update how things are going: what's
working, what isn't, and how does it change your timeline?  Have the
middle person alphabetically submit this one page update.  You'll also need to have your first submission submitted.

Final Presentation
======================

The final presentation will be virtual (uploading a video).  In
the final presentation you will:

* Explain what you did

* Who did what

* What challenges you had

* Review how well you did (based on the competition)

* Provide an error analysis.  An error analysis must contain examples from the
  development set that you get wrong.  You should show those sentences
  and explain why (in terms of features or the model) they have the
  wrong answer.  You should have been doing this all along as you
  derive new features, but this is your final inspection of
  your errors. The feature or model problems you discover should not
  be trivial features you could add easily.  Instead, these should be
  features or models that are difficult to correct.  An error analysis
  is not the same thing as simply presenting the error matrix, as it
  does not inspect any individual examples.

* The linguistic motivation for your features / how your wrote the questions.  This is a
  computational linguistics class, so you should give precedence to
  features / techniques that we use in this class (e.g., syntax,
  morphology, part of speech, word sense, etc.).  Given two features
  that work equally well and one that is linguistically motivated,
  we'll prefer the linguistically motivated one.

* Presumably you did many different things; how did they each
  individually contribute to your final result?

Each group has 10 minutes to deliver their presentation. Please record the video, and upload it to Google Drive, and include the link in your writeup submission.

System Submission
======================

You must submit a version of your system by May 10.  It may not be perfect, but this what the question writing teams will use to test their results.


Project Writeup
======================

By May 17, submit your project writeup explaining what
you did and what results you achieved.  This document should
make it clear:

* Why this is a good idea
* What you did
* Who did what
* Whether your technique worked or not

For systems, please do not go over 2500 words unless you have a really good reason.
Images are a much better use of space than words, usually (there's no
limit on including images, but use judgement and be selective).

For question writing, you have one page (single spaced, two column) per question plus a two page summary of results.

Grade
======================

The grade will be out of 25 points, broken into five areas:

* _Presentation_: For your oral presentation, do you highlight what
  you did and make people care?  Did you use time well during the
  presentation?

* _Writeup_: Does the writeup explain what you did in a way that is
  clear and effective?

* _Technical Soundness_: Did you use the right tools for the job, and
  did you use them correctly?  Were the relevant to this class?

* _Effort_: Did you do what you say you would, and was it the right
  ammount of effort.

* _Performance_: How did your techniques perform?

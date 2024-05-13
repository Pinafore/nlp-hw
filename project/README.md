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

We will use HuggingFace's leaderboard
(https://huggingface.co/spaces/umdclip/grounded_qa_leaderboard). Here, you submit your QA model and assess how your QA
model did compared to others. The assessment will take place by
testing your QA model on *different* data than what we'll use for the
final session.
Your score will be visible on the leaderboard. Your
goal is to rank the highest in terms of the [Buzz confidence](https://docs.google.com/document/d/1IgGSngNlIv5ZvcTh2_3kJqQKdcKDgz41qDZtWDWkY3k/edit?usp=sharing): you want
to have high confidence when you're right and low confidence when
you're wrong.


Writing Questions
==================

Alternatively, you can also *write* 50 adversarial questions that
challenge modern NLP systems. These questions must be diverse in the
subjects asked about, the skills computers need to answer the
questions, and the entities in those questions. Remember that your questions should be *factual* and
*specific* enough for humans to answer, because your task is to stump
the computers relative to humans!

In addition to the raw questions, you will also need to create citations describing:
* Why the question is difficult for computers: include citations from the NLP/AI/ML literature
* Why the information in the question is correct: include citations from the sources you drew on the write the question
* Why the question is interesting: include scholarly / popular culture artifacts to prove that people care about this
* Why the question is pyramidal: discuss why your first clues are harder than your later clues

However, you'll be scored on the same metric as the system: you want
humans to buzz correctly before the computer has the correct answer
with high confidence.

**Category**

We want questions from many domains such as Art, Literature, Geography, History,
Science, TV and Film, Music, Lifestyle, and Sport. The questions
should be written using all topics above (5 questions for each
category and 5 more for the remaining categories). Indicate in your
writeup which category you chose to write on for each question.


Art:

* Questions about works: Mona Lisa, Raft of the Medussa

* Questions about forms: color, contour, texture

* Questions about artists: Picasso, Monet, Leonardo da Vinci

* Questions about context: Renaissance, post-modernism, expressionism, surrealism


Literature: 

*	Questions about works: novels (1984), plays (The Lion and the Jewel), poems (Rubaiyat), criticism (Poetics)

*	Questions about major characters or events in literature: The Death of Anna Karenina, Noboru Wataya, the Marriage of Hippolyta and Theseus

*	Questions about literary movements (Sturm und Drang)

*	Questions about translations

*	Cross-cutting questions (appearances of Overcoats in novels)

*	Common link questions (the literary output of a country/region)


Geography: 

*	Questions about location: names of capital, state, river

*	Questions about the place: temperature, wind flow, humidity


History: 

*	When: When did the First World war start? 

*	Who: Who is called Napoleon of Iran? 

*	Where: Where was the first Summer Olympics held?

*	Which: Which is the oldest civilization in the world?


Science: 

*	Questions about terminology: The concept of gravity was discovered by which famous physicist?

*	Questions about the experiment

*	Questions about theory: The social action theory believes that individuals are influenced by this theory.


TV and Film: 

*	Quotes: What are the dying words of Charles Foster Kane in Citizen Kane?

*	Title: What 1927 musical was the first "talkie"?

*	Plot: In The Matrix, does Neo take the blue pill or the red pill?


Music: 

*	Singer: What singer has had a Billboard No. 1 hit in each of the last four decades?

*	Band: Before Bleachers and fun., Jack Antonoff fronted what band?

*	Title: What was Madonna's first top 10 hit?

*	History: Which classical composer was deaf?


Lifestyle: 

*	Clothes: What clothing company, founded by a tennis player, has an alligator logo?

*	Decoration: What was the first perfume sold by Coco Chanel?


Sport: 

*	Known facts: What sport is best known as the ‘king of sports’?

*	Nationality: What’s the national sport of Canada?

*	Sport player: The classic 1980 movie called Raging Bull is about which real-life boxer?

*	Country: What country has competed the most times in the Summer Olympics yet hasn’t won any kind of medal?


**Diversity** 

Other than category diversity, if you find an ingenious way of writing questions about underrepresented countries, you will get bonus points (indicate which questions you included the diversity component in your writeup). You may decide which are underrepresented countries with your own reasonable reason (etc., less population may indicate underrepresented), but make sure to articulate this in your writeup. 

* Run state of the art QA systems on the questions to show they struggle, give individual results for each question and a summary over all questions

For an example of what the writeup for a single question should look like, see the adversarial HW:
https://github.com/Pinafore/nlp-hw/blob/master/adversarial/question.tex


This is a Final
------------

While writing questions is designed to be fun (and it is), this is
still the place for you to showcase what you've learned about natural
language processing.  You should review the entire course's material
not just for ideas of what topics to write about but also for
techniques of how to write the questions.  For example:

* The role of negation specifically and syntax more generally and how
  that can help you
  
* The kind of data that QA systems (and NLP systems generally) are
  trained on [i.e., if the information is in Wikipedia, you're going
  to have a bad time]
  
* The role of attention in LLMs, and how that's used to generate text
  [i.e., if the model isn't attending to a token, it's not going to
  get used in answering a question]
  
* How you can use tf-idf as a proxy for attention for QA-like tasks
  [if it's low tf-idf, it's likely not going to get high attention
  ... use scene descriptions and descriptions of visual elements
  whenever you can]
  
While all of the course's lectures could help you, some suggestions of
material to review:

* [How stump a computer](https://www.youtube.com/watch?v=6oZCIOBiSaI)
* [Good questions](https://www.youtube.com/watch?v=uVcPlJu-JCM)
* [Bad questions](https://youtu.be/LKQVJgj5ffg)
* [Previous final](https://www.youtube.com/watch?v=dyaR7zT_KKg)
* [QA Datasets](https://youtu.be/p8tnM1_waQ8)

Proposal
==================

The project proposal is a one page PDF document that describes:

* Who is on your team (team sizes can be between three and six
  students, but six is really too big to be effective; my suggestion
  is that most groups should be between four or five).

* What techniques you will explore 

* Your timeline for completing the project (be realistic; you should
  have your first submission in a week or two)

Submit the proposal on Gradescope, but make sure to include all group
members.  If all group members are not included, you will lose points.  Late days cannot be used on this
assignment.

Milestone 1
====================== 

You'll have to update how things are going: what's
working, what isn't, and how does it change your timeline?  How does it change your division of labor?

*Question Writing*: You'll need to have answers selected for all of
your questions and first drafts of at least 15 questions.  This must
be submitted as a JSON file so that we run computer QA systems on it.

You do not need to submit the PDF writeup for this milestone, but it
will be required for the next one.

*Project*: Submit a PDF updating on your progress to Gradescope.

If all team members are not on the submission, you will lose points.

Milestone 2
===================

As before, provide an updated timeline / division of labor, provide your intermediary results.  

*Question Writing*: You'll need to have reflected the feedback from
 the first questions and completed a first draft of at least 30
 questions.

You'll also need machine results to your questions and an
 overall evaluation of your human/computer accuracy.  Unlike the
 previous submission, this will require submitting a PDF detailing
 your sources and current status.

 The goal for the PDF is to show that you've started on the work of justifying your question writing with sources.  Not every question needs to have a complete writeup, but the questions that are furthest along should have their writeup started.

Then your cover sheet can talk about your process and progress thus far.

In other words, it should be a first draft of your final writeup with any concerns or places you need help / guidance.
 
*Project*: You'll need to have a made a submission to the leaderboard with a working system.

Submit a PDF updating on your progress.

Final Presentation
======================

The final presentation will be virtual (uploading a video).  In
the final presentation you will:

* Explain what you did

* Who did what.  For example, for the question writing project a team of five people might write: A wrote the first draft of questions.  B and C verified they were initially answerable by a human.  B ran computer systems to verify they were challenging to a computer.  C edited the questions and increased the computer difficulty.  D and E verified that the edited questions were still answerable by a human.  D and E checked all of the questions for factual accuracy and created citations and the writeup.

* What challenges you had

* Review how well you did (based on the competition or your own metrics).  If you do not use the course infrastructure to evaluate your project's work, you should talk about what alternative evaluations you used, why they're appropriate/fair, and how well you did on them.

* Provide an error analysis.  An error analysis must contain examples from the
  development set that you get wrong.  You should show those sentences
  and explain why (in terms of features or the model) they have the
  wrong answer.  You should have been doing this all along as you
  derive new features, but this is your final inspection of
  your errors. The feature or model problems you discover should not
  be trivial features you could add easily.  Instead, these should be
  features or models that are difficult to correct.  An error analysis
  is not the same thing as simply presenting the error matrix, as it
  does not inspect any individual examples.  If you're writing questions, talk about examples of questions that didn't work out as intended.

* The linguistic motivation for your features / how your wrote the questions.  This is a
  computational linguistics class, so you should give precedence to
  features / techniques that we use in this class (e.g., syntax,
  morphology, part of speech, word sense, etc.).  Given two features
  that work equally well and one that is linguistically motivated,
  we'll prefer the linguistically motivated one.

* Presumably you did many different things; how did they each
  individually contribute to your final result?

Each group has 10 minutes to deliver their presentation. Please record
the video, and upload it to Google Drive (or somewhere else where
course staff can watch it), and include the link in your writeup submission.

Final Question Submission
======================

Because we need to get the questions ready for the systems, upload
your raw questions on May 10 in json format on Gradescope.  This doesn't include the citations or
other parts of the writeup.

System Submission
======================

You must submit a version of your system by May 12. It may not be
perfect, but this what the question writing teams will use to test
their results.

Your system should be uploaded directly to the HuggingFace
leaderboard.

Project Writeup and JSON file
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

For question writing, you have one page (single spaced, two column) per question plus a two page summary of results. Talk about how you organized the question writing, how you evaluated the questions, and a summary of the results.  Along with your writeup, turn in a json including the raw text of the question and answer and category. The json file is included in this directory. Make sure your json file is in the correct format and is callable via below code. Your submission will not be graded if it does not follow the format of the example json file.  

```
with open('path to your json file', 'r') as f:
    data = json.load(f)
```

Grade 
======================

The grade will be out of 25 points, broken into five areas:

* _Presentation_: For your oral presentation, do you highlight what
  you did and make people care?  Did you use time well during the
  presentation?

* _Writeup_: Does the writeup explain what you did in a way that is
  clear and effective?

The final three areas are different between the system and the questions.

|    |      System      |  Questions |
|----------|:-------------:|------:|
| _Technical Soundness_ |  Did you use the right tools for the job, and did you use them correctly?  Were they relevant to this class? | Were your questions correct and accurately cited. |
| _Effort_ |  Did you do what you say you would, and was it the right ammount of effort.  | Are the questions well-written, interesting, and thoroughly edited? |
| _Performance_ | How did your techniques perform in terms of accuracy, recall, etc.? | Is the human accuracy substantially higher than the computer accuracy? |

All members of the group will receive the same grade.  It's impossible
for the course staff to adjudicate Rashomon-style accounts of who did
what (I know that the book is called "In a Grove", but most people are
familiar with the movie called "Rashomon", which took the story of "In
a Grove" but took the title of another story), and the goal of a group
project is for all team members to work together to create a cohesive
project that works well together.  While it makes sense to divide the
work into distinct areas of responsibility, at grading time we have
now way to know who really did what, so it's the groups responsibility
to create a piece of output that reflects well on the whole group.


FAQ
======================

*Q:* What system will you use to evaluate our questions?  It's really
hard to write adversarial questions against the best LLMs.

*A:* We will test your questions with all of the student submitted
 systems and with a handful of our own systems (including GPT and
 Claude).  For the “full” evaluation, we’re going to have multiple
 agents (both human and computer) and we’re going to compare the
 difference between human and computer accuracy.  For “milestone”
 evaluations, we’re just going to do random checks.  For models that are
 stochastic, we’ll indeed take multiple runs.  Thus, you should aim
 for questions that models consistently fail to answer … we won’t take
 the “best” result.

*Q:* Do references count toward the page count?

*A:* No.


*Q.* Does everyone need to be in the presentation?  

*A.* Nope, you can use your time however you want.  You can have
everyone in it or you can just have one person do the whole
presentation.  You also don't need to use the full time if you don't
feel like you need it.

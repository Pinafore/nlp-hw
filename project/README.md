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

Alternatively, you can also *write* 50 adversarial questions that
challenge modern NLP systems. These questions must be diverse in the
subjects asked about, the skills computers need to answer the
questions, and the entities in those questions. These questions should
be submitted in two versions, pyramidal and non-pyramidal (so total of
100). Pyramidal questions should have qualifications to make them
PYRAMIDAL (etc., each sentence should be identifiable, and the
sentence order should be pyramidal), and the non-pyramidal questions
should have good coverage of the pyramidal question that you've
written. Remember that your questions should be *factual* and
*specific* enough for humans to answer, because your task is to stump
the computers relative to humans!

In addition to the raw questions, you will also need to create citations describing:
* Why the question is difficult for computers: include citations from the NLP/AI/ML literature
* Why the information in the question is correct: include citations from the sources you drew on the write the question
* Why the question is interesting: include scholarly / popular culture artifacts to prove that people care about this
* Why the question is pyramidal: discuss 
* And an interesting non-pyramidal question that could be used for something like OQL

**Example of OQL Questions**

Q. France has seen a series of strikes and demonstrations in its capital city, after proposals to raise what from the age of 62 to 64? 
A. RETIREMENT AGE 

Q. Which 2004 teen romantic comedy tells of a girl, played by Hilary Duff, forced to work in her stepmother’s diner after the death of her father? She begins an online relationship with 'Nomad' who is later revealed to be the most popular boy in school, Austin, played by Chad Michael Murray.
A. CINDERELLA STORY

For more examples, refer to https://quizcentral.net/quizzes/OQL-UK-SEASON-9-CUP-QUIZ-ROUND-3.pdf

**Category**

OQL contains topics such as Art, Literature, Geography, History,
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

Each group has 10 minutes to deliver their presentation. Please record the video, and upload it to Google Drive, and include the link in your writeup submission.

System Submission
======================

You must submit a version of your system by May 10. It may not be perfect, but this what the question writing teams will use to test their results.

Your system should be sent directly to yysung53@umd.edu in zip files, including the correct dependencies and a working inference code. Your inference code should run successfully in the root folder (extracted from zip folder) directory with the command:

```
> python3 inference.py --data=evaluation_set.json 

```

The input will in the form of evaluation_set.json file which is the same format with the file that adversarial question writing team submits. The output format should also be in string. 

If you have any notes or comments that we should be aware of while running your code, please include them in the folder as .txt file. Also, dependency information should be included as .txt file. 

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

All members of the group will receive the same grade.  It's impossible for the course staff to adjudicate Rashomon-style accounts of who did what, and the goal of a group project is for all team members to work together to create a cohesive project that works well together.  While it makes sense to divide the work into distinct areas of responsibility, at grading time we have now way to know who really did what, so it's the groups responsibility to create a piece of output that reflects well on the whole group.

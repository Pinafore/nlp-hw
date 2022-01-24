Homework 0: Getting Started [100 pts]
--

The goal of this homework is to help you get started with submitting QA based programming assignments and getting used to leaderboards over Gradescope.

Every individual must submit this homework.


The homework has two parts:

### A. Fault in our assistants [50 pts]

You are required to answer following two equally weighed questions over Gradescope:

1. Submit a question that Siri, Alexa, Cortana, Google Assistant, etc. got pretty wrong. [25 pts]
2. Discuss why it should have been answered differently and if there are variants of the question that it does answer correctly. [25 pts]


### B. Say hello to QA! [50 pts]

This is a programming assignment where you will code up a very simple rule based QA system that outputs a fixed answer for a given rule.

This will help you get used to submitting QA systems on Gradescopes, make you familiar with leaderboards.


#### Rules:
If the question starts with the word:
* “who”, it answers “Hatschepsut”
* “when”, it answers “1215”
* “where”, it answers “Mount Meru”
* “what”, it answers “Tofu”
* “why”, it answers "42"
* Otherwise, answer “I don’t know”

The input question can be assumed to be case insensitive, however the output is case sensitive.

#### How to turn in my homework ?

- You need to submit a zip file that has `qa_hw0.py` source file at the root level.
- The file must have a class `SimpleQARunner` that has
  -  a default empty constructor, and 
  -  a method `execute_query(question_text: str) -> str`
  -  It must consume a string that is a case insensitive english language sentence and outputs a case sensitive answer as per above defined rules.

We have provided you with the stubs in the file `qa_hw0.py` in this directory which you can populate with the required piece of code before submitting on Gradescope.

For future homeworks you will be asked to wrap a full fledged QA system inside a similar class that our autograder will instantiate to run against the test cases, and display your scores over the leaderboard.

Happy Coding!
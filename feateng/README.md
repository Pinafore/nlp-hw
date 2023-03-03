Feature Engineering
=

Before, you built a logistic regression system from scratch and tested it on
how well it could predict if an answer was correct.

You're going to continue to improve the accuracy of such a system by creating
new features.

You will improve the classification by extracting useful information from
the guesses and generate better features for input into the *logistic
regression* classifier to do a better job of selecting whether a guess to a
question is correct.

NOTE: Because the goal of this assignment is feature engineering, not
classification algorithms, you may not change the underlying algorithm. You
can change add to the guessed answers (e.g., to create a new feature), but you
may not swap out the class that's generating classes nor can you change the
classifier.

This assignment is structured in a way that approximates how classification
works in the real world: features are typically underspecified (or not
specified at all). You have to articulate the features you
need. You then compete against others to provide useful predictions.

It may seem straightforward, but do not start this at the last minute. There
are often many things that go wrong in testing out features, and you'll want
to make sure your features work well once you've found them.

Getting Started
-

We'll use some more packages than we had before:

    pip3 install unidecode nltk sklearn python-baseconv spacy
    python -m nltk.downloader stopwords

You'll also need to create a directory for the models you'll be
creating

     mkdir -p models

How to add a feature?
-

First, get an idea of what you want to do.  After training the classifier,
look at your predictions on the dev set and see where they're going wrong.

1.  To add a feature, you need to create a new subclass of the Feature class
in ``features.py``.  This is important so that you can try out different
features by turning them on and off with the command line.

2.  Add code to instantiate the feature in ``params.py``.

3.  (Optionally) Change the API to feed in more information into the feature
generation process.  This would be necessary to capture temporal dynamics or
use, say, information from Wikipedia:
https://drive.google.com/file/d/1-AhjvqsoZ01gz7EMt5VmlCnVpsE96A5n/view?usp=sharing

To walk you through the process, let's create a new feature that encodes how
often the guess appeared in the training set.  The first step is to define the
class in ``features.py``.

	class FrequencyFeature:                       
	    def __init__(self, name):                 
		from buzzer import normalize_answer   
		self.name = name                      
		self.counts = Counter()               
		self.normalize = normalize_answer     

	    def add_training(self, question_source):    
		import json                 
		with open(question_source) as infile:                   
			questions = json.load(infile)                       
			for ii in questions:                                
			    self.counts[self.normalize(ii["page"])] += 1    

	    def __call__(self, question, run, guess):                               
		yield ("guess", log(1 + self.counts[self.normalize(guess)]))          
    
Then the class needs to be loaded.  This happens in ``params.py``.  Now you can
add the feature name to the command line to turn it on.

    for ff in flags.features:
        if ff == "Length":
            from features import LengthFeature
            feature = LengthFeature(ff)
            buzzer.add_feature(feature)

        if ff == "Frequency":                                  
            from features import FrequencyFeature              
            feature = FrequencyFeature(ff)                     
            feature.add_training("../data/qanta.buzztrain.json")
            buzzer.add_feature(feature)                        


Before we try it out, we need to know what our baseline is.  So let's see how
it did *without* that feature.


    mkdir -p models
    python3 buzzer.py --guesser_type=GprGuesser --limit=50 \
      --question_source=json --GprGuesser_filename=../models/GprGuesser \
      --questions=../data/qanta.buzztrain.json --buzzer_guessers GprGuesser

After training the classifer, you should see something that looks like this:

	Loaded 1660 entries from cache
	INFO:root:Made 0 new queries, saving to ../models/GprGuesser
	INFO:root:Made 0 new queries, saving to ../models/GprGuesser
	INFO:root:Made 0 new queries, saving to ../models/GprGuesser
	INFO:root:Made 0 new queries, saving to ../models/GprGuesser
	INFO:root:Made 0 new queries, saving to ../models/GprGuesser
	/home/jbg/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
	STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

	Increase the number of iterations (max_iter) or scale the data as shown in:
	https://scikit-learn.org/stable/modules/preprocessing.html
	Please also refer to the documentation for alternative solver options:
	https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
	n_iter_i = _check_optimize_result(
	INFO:root:Made 0 new queries, saving to ../models/GprGuesser
	Ran on 50 questions of 50

(The warning about convergence is okay; hopefully it will converge better with
more features!)

Now you need to evaluate the classifier.  The script eval.py will run the classifier on all of your data and then record the outcome.  There are several things that could happen:
 * _best_: Guess was correct, Buzz was correct
 * _timid_: Guess was correct, Buzz was not
 * _aggressive_: Guess was wrong, Buzz was wrong
 * _waiting_: Guess was wrong, Buzz was correct

 Now, both "best" and "waiting" are *correct*, but obviously "best" is best.  It's important to know what kind of examples contribute to each of these outcomes, so eval samples a subset for each of these and prints them and their features out.  

        python3 eval.py --guesser_type=GprGuesser --TfidfGuesser_filename=models/TfidfGuesser --limit=25 --question_source=json --questions=../data/qanta.buzzdev.json --logging_file=buzzer.log --buzzer_guessers GprGuesser --GprGuesser_filename=../models/GprGuesser

    [snip ...]

    answer 0.01
    ==================

               guess: Donald Davidson (philosopher)
              answer: Donald_Davidson_(philosopher)
                  id: 93152
    GprGuesser_guess: Donald Davidson (philosopher)
      Gpr_confidence: -0.5728
     consensus_guess: Donald Davidson (philosopher)
     consensus_count: 1
         Length_char: 6.6708
        Length_guess: 3.4012
         Length_word: 4.8040
                text: This thinker wrote that "framework theories" cannot make sense of
                      radio host Goodman Ace's malapropisms. This philosopher argued that an
                      actor's "pro-attitude" must be part of the "primary reason" that
                      causes an action. This author of "A Nice Derangement of Epitaphs"
                      proposed using Tarski's semantic theory of truth as the core for a
                      "theory of meaning," though he later claimed "there is no such thing
                      as a language." He included the "principle of charity," which assumes
                      that another speaker has true beliefs, in a method for understanding
                      unfamiliar speech "from scratch." His alternative to mind-body dualism
                      held that no natural laws connect physical events with mental events.
                      For 10 points, name this American philosopher who devised "radical
                      interpretation" and anomalous monism.
    --------------------
               guess: Frigg
              answer: Frigg
                  id: 93171
    GprGuesser_guess: Frigg
      Gpr_confidence: -6.0745
     consensus_guess: Frigg
     consensus_count: 1
         Length_char: 6.5539
        Length_guess: 1.7918
         Length_word: 4.8442
                text: Most scholars identify this deity with a figure named Saga who dwells
                      in Sokkvabekk. Along with a servant, this deity helped to heal the
                      horse of Phol. Hlin and Syn serve this figure, who told the women of
                      Winnili to cover their faces with hair, thus helping to found the
                      Lombards. Two other servants of this deity, who ride the horse
                      Hofvarpnir and carry shoes respectively, are Gna and Fulla. At the
                      hall Fensalir, this goddess spins the clouds on a loom. Loki accused
                      this goddess of having affairs with Vili and Ve. After this goddess
                      sent Hermod on a mission to Hel, the giantess Thokk refused to weep
                      for her dead son because this goddess failed to get an oath from
                      mistletoe to remain harmless.

It's only answering two questions correctly.  And it's waiting a lot when the
answer is correct.  So let's try it with the feature turned on (don't forget
to retrain the model).

    python3 buzzer.py --guesser_type=GprGuesser --limit=50 \
    --question_source=json --GprGuesser_filename=../models/GprGuesser \
    --questions=../data/qanta.buzztrain.json --buzzer_guessers GprGuesser \
    --features Length Frequency

    python3 eval.py --guesser_type=GprGuesser \
    --TfidfGuesser_filename=models/TfidfGuesser --limit=25 \
    --question_source=json --questions=../data/qanta.buzzdev.json \
    --logging_file=buzzer.log --buzzer_guessers GprGuesser \
    --GprGuesser_filename=../models/GprGuesser --features Length Frequency


    ==================
    answer 0.06
    ==================


               guess: The Awakening (Chopin novel)
              answer: The_Awakening_(Chopin_novel)
                  id: 93160
    GprGuesser_guess: The Awakening (Chopin novel)
      Gpr_confidence: -0.4093
     consensus_guess: The Awakening (Chopin novel)
     consensus_count: 1
         Length_char: 6.4036
        Length_guess: 3.3673
         Length_word: 4.6052
     Frequency_guess: 3.4657
                text: This character faintheartedly commits herself to improving her studies
                      after a night of reading Emerson alone in her house, and hushes Victor
                      when he begins singing "Ah! Si tu savais!" While talking to a friend,
                      she declares that she would give up the "unessential things" for her
                      children, but she wouldn't give herself up. Doctor Mandelet advises
                      this character's husband to permit her whims, which include moving
                      into a "pigeon house" outside of her house on Esplanade Street. This
                      mother of Raoul and Etienne watches Adele Ratignolle give birth on her
                      last night alive, and romances Alcee Arobin and
    --------------------
               guess: Athol Fugard
              answer: Athol_Fugard
                  id: 93163
    GprGuesser_guess: Athol Fugard
      Gpr_confidence: -6.3761
     consensus_guess: Athol Fugard
     consensus_count: 1
         Length_char: 6.5568
        Length_guess: 2.5649
         Length_word: 4.8903
     Frequency_guess: 3.4965
                text: In a play by this man, one title character counts the bruises caused
                      by the other title character, who accuses her of looking behind her to
                      find a dog on the road. This author also wrote a play in which two men
                      stage an impromptu performance of Sophocles' Antigone after getting
                      off their shifts as prison workers. This man created a teenager who
                      debates the idea of a "Man of Magnitude" to aid his composition for an
                      English class, as well two campers who take in an old man who does not
                      speak English. A third play by this author of Boesman and Lena and The
                      Island takes place just as the title antagonist's father is coming
                      home from the hospital, which prompts him to be cruel to Sam and
                      Willie, his
    --------------------
               guess: Athol Fugard
              answer: Athol_Fugard
                  id: 93163
    GprGuesser_guess: Athol Fugard
      Gpr_confidence: -6.3834
     consensus_guess: Athol Fugard
     consensus_count: 1
         Length_char: 6.6908
        Length_guess: 2.5649
         Length_word: 4.9972
     Frequency_guess: 3.4965
                text: In a play by this man, one title character counts the bruises caused
                      by the other title character, who accuses her of looking behind her to
                      find a dog on the road. This author also wrote a play in which two men
                      stage an impromptu performance of Sophocles' Antigone after getting
                      off their shifts as prison workers. This man created a teenager who
                      debates the idea of a "Man of Magnitude" to aid his composition for an
                      English class, as well two campers who take in an old man who does not
                      speak English. A third play by this author of Boesman and Lena and The
                      Island takes place just as the title antagonist's father is coming
                      home from the hospital, which prompts him to be cruel to Sam and
                      Willie, his black servants. For 10 points, name this South African
                      playwright of "Master Harold"...and the Boys.
    --------------------
               guess: Frigg
              answer: Frigg
                  id: 93171
    GprGuesser_guess: Frigg
      Gpr_confidence: -6.0745
     consensus_guess: Frigg
     consensus_count: 1
         Length_char: 6.5539
        Length_guess: 1.7918
         Length_word: 4.8442
     Frequency_guess: 2.8904
                text: Most scholars identify this deity with a figure named Saga who dwells
                      in Sokkvabekk. Along with a servant, this deity helped to heal the
                      horse of Phol. Hlin and Syn serve this figure, who told the women of
                      Winnili to cover their faces with hair, thus helping to found the
                      Lombards. Two other servants of this deity, who ride the horse
                      Hofvarpnir and carry shoes respectively, are Gna and Fulla. At the
                      hall Fensalir, this goddess spins the clouds on a loom. Loki accused
                      this goddess of having affairs with Vili and Ve. After this goddess
                      sent Hermod on a mission to Hel, the giantess Thokk refused to weep
                      for her dead son because this goddess failed to get an oath from
                      mistletoe to remain harmless.


Okay, so that helped!

                         Frequency_guess: 0.3641
                   GprGuesser_confidence: 0.0651
                             Length_char: 0.4994
                            Length_guess: -0.2049
                             Length_word: 0.6408
    Accuracy: 0.73  Buzz ratio: 16.00

At the end of the eval script, you can see the features that it's using, the
overall accuracy, and the ratio of correct buzzes to incorrect buzzes (should
be positive).

What Can You Do?
-

You can:
* Add features (e.g., to params.py)
* Change feature representations (e.g., features.py)
* Exclude data 
* Add data

What Can't You Do?
-
Change the static guesses or use a different classifier (buzzer in this lingo).

How to start
-
1. Remind yourself how to run the sklearn logistic regression (logistic_buzzer.py)
2. Add a simple feature to the training data generated by gpr_guesser.py 
3. See if it increases the accuracy on held-out data when you run logistic regression (eval.py) or on the leaderboard
4. Rinse and repeat!


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
reporting the corresponding metrics for each attempt.  You are expected to pay more
attention to what is going on with the data and take a data-driven
approach to feature engineering.

How to Turn in Your System
-
* ``features.py``: This file includes an implementation of your new features.
* ``params.py``: This instantiates your new features.  Modify this so that the
set of your best features runs by *default*.  
* **Custom Training Data** (If you used additional training data beyond the Wikipedia pages, upload that as well
    * (OR) If either any of your files are >100MB, please submit a shell
    script named ``gather_resources.sh`` that will retrieve one or both of the
    files above programatically from a public location (i.e a public S3
    bucket).
* The LogisticBuzzer.model.pkl file and LogisticBuzzer.featurizer.pkl file created by training the classifier.
* ``analysis.pdf``: Your **PDF** file containing your feature engineering
analysis.

Turn in the above files as usual via Gradescope, where we'll be using the
leaderboard as before.  However, the position on the leaderboard will count
for more of your grade.

FAQ
-----------------

*Q:* Can I modify buzzer.py so that I can use the history of guesses in a
 question?

*A:* Yes.  If you do that, make sure to upload the file.  We will replace the
 default version of buzzer.py with your new submission.

*Q:* Can I use the <INSERT NAME HERE> package?

*A:* Clear it first on Piazza.  We'll provide spacy and nltk for sure (along
 with all of the packages already used in this homework).  We
 won't allow packages that require internet access (e.g., wikipedia).  We
 don't have anything against Wikipedia (we provide this json file so you can
 use it), but we don't want to get our IP
 address banned.

*Q:* Sometimes the guess is correct but it isn't counted that way.  And
 sometimes a wrong answer is counted as correct.

*A:* Yes, and we'll cover this in more detail later in the course.  For now,
 this is something we'll have to live with.

*Q:* What if I get the error that ``GprGuesser`` has no attribute 'predict'?

*A:* This means that you're running it on a guesser result that hasn't been
 cached or that it can't find the cache file.  Make sure the path is correct,
 and use the limit option to only process a handful of examples.

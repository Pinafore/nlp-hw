
Big Picture
=====

While we've trained models from scratch, often the best place to start is with
an existing model.  In this homework, we're going to do two things: 1) use
prompting to improve the outputs of a black box model for the guessing task 2)
fine tune a large(ish) open weight model so that it can do the same take that
you saw in your feature engineering homework: predicting whether an answer to
a question is correct or not.

This homework is worth 35 points, but distributed over three people.

General Setup
=====

As usual, you may need to install athe required Python environment (there's a lot this time).

    ./venv/bin/pip3 install -r requirements.txt

We'll also be using ollama:

     curl -fsSL https://ollama.com/install.sh | sh

You may need to install as ``sudo``.  After installing ollama, you'll want to
download the model you'll be using.  ``gemma3:4b`` will run on most hardware.
Things will be much better if you have a GPU to use.  The TAs will be making
Nexus accounts available if you do not have your own hardware.

For running on Nexus, you do not have sudo, so you'll need to do a (local
install)[https://medium.com/@abdulsalamelelu/executing-ollama-on-hpc-without-sudo-access-7eb6217e6fcb],
which will be slightly more complicated.

What you Have to Do (Guesser)
=====

To get things set up, you'll need to train a **TfIdfGuesser**.  This was the
subject of a previous homework, so just use your previous solution for that.
The quality of this element matters a lot for the RAG step, so you may want to
spend some time tuning it if you didn't do it for the previous homework.
Recall is much more important than precision.

	.venv/bin/python3 guesser.py --guesser_type=Tfidf --question_source=gzjson --questions=../data/qanta.guesstrain.json.gz --logging_file=guesser.log

The first priority is to get it running and sending requests to Ollama.  The
actual optimization process can take a while (it depends on your hardware), so
don't leave this until the last minute.  You also cannot request a Nexus
account later than a week before the due date, so if you need one, make sure
to ask ASAP.

Some suggestions on what you could do (non-exhaustive, and you can try other stuff):
- Change the pipeline to explicitly determine the lexical answer type
- Run multiple optimizations to separately tune the query formulation, guessing, and confidence estimation [if you do this, make sure you don't overfit on the validation data ... make sure to divide it up]
- Add more RAG inputs (e.g., Wikipedia)
- Tune the RAG outputs (right now, it's just the retrieved sentence, more context might be helpful)
- Create explicit intermediate results that could help calibration (e.g., number of RAG hits that match, overlap between RAG and output, does the guess appear in the question text)

But whatever you do, the default main function of `ollama_guesser.py` will
train a teleprompter for you.  At the moment, the code is everything I had in
my lecture.  You may want to do more or less than what I did.

   ./venv/bin/python3 ollama_guesser.py

What you Have to Do (Buzzer)
=====

After understanding the code, you can get down to coding:

* To make sure that this is actually efficient, you will need to freeze the
  model's original parameters.  Set the `requires_grad` for all of the base
  model parameters to `False`.  You need to do this in the
  `initialize_base_model` code.  *Do not overlook this, as it will work
  without this change but will be very slow*.  The first time you run the code will take a little bit longer because it needs
to download the DistillBERT model.

      ./venv/bin/python3 lorabert_buzzer.py --questions=../data/qanta.buzztrain.json.gz --secondary_questions=../data/qanta.buzzdev.json.gz --buzzer_guessers=Tfidf --load=True --buzzer_type='lorabert' --limit=100
      config.json: 100%|████████████████████████████████████████████| 483/483 [00:00<00:00, 7.18MB/s]
      model.safetensors: 100%|████████████████████████████████████| 268M/268M [00:04<00:00, 64.1MB/s]

This will go faster afterward.

* You will need to define the parameter matrices for the LoRA layer in the
  `LoRALayer` class `__init__` function and then use them to compute a delta
  in the `forward` function.

* Likewise, you will need to add a `LoRALayer` component to the `LinearLoRA`
  class and change its `forward` function to use that delta in its forward
  function.  (I realize this could have been one class, but this makes testing
  easier... it also makes it possible to have more LoRA adaptations beyond
  adapting just linear layers.)

* Now that we have the tools for changing some layers, we now need to add them
  to the frozen model we created in `initialize_base_model` in the `add_lora`
  function.  You will probably want to create a (partial
  object)[https://docs.python.org/3/library/functools.html#partial-objects].  

* The only changes you should have to make in LoRABertBuzzer are to add
  methods to load and save the buzzer to a file.

* Run adaptation on some data (use `limit` if you don't have a GPU).  This is
  more of a proof-of-concept, and you don't need great accuracy to satisfy the
  requirements of the homework (but loss should go down and accuracy should
  improve with more data).

* The command line above uses the default tf-idf guesser, but you'll want to
  replace it with the DSPy-based guesser you've optimized.  E.g., something like.

           ./venv/bin/python3 lorabert_buzzer.py --questions=../data/qanta.buzztrain.json.gz --secondary_questions=../data/qanta.buzzdev.json.gz --buzzer_guessers=Ollama --load=True --buzzer_type='lorabert' --limit=100

  This loads the save DSPy model from models/guesser.json ... if you saved it
  to a different file, you'll need to change the `ollama_guesser_filename`
  flag.

Good Enough Solution
=====

To have a good enough solution, you must both
1. improve the Guesser to have a higher recall and precision over the working implementation you've been given **conditioned on the underlying black box Muppet Model**.
2. improve the expected wins of the finetuned over either logistic regression with just the confidence feature and length or logistic regression with just the confidence feature and the guess.  (This will depend on your Guesser, obviously)

What to Submit
=====
We create an assignment where you will submit your trained guesser and buzzer. Please submit following files to Gradescope.

* Your `analysis.pdf` file (if you don't go beyond the "Good
Enough", you must at least establish your baseline values).

* Your `lorabert.model` file (where you did your finetuning).

* Your `dspy.json` model (the final prompts found via teleprompting).

* Your `parameters.py` with appropriate defaults to run your model

* Your `ollama_guesser.py` with any changes to the prompt definitions

* Your `lorabert_buzzer.py` with completed code

* Your `TfidfGuesser.answers.pkl`, `TfidfGuesser.questions.pkl`, `TfidfGuesser.tfidf.pkl` and `TfidfGuesser.vectorizer.pkl` (a tfidf model)

For this HW, since running ollama on Gradescope isn't possible for us, we will grade your submission manually. The autograder will only check whether all required files are uploaded. We will run the submissions on 
* 11/20-11/25 at 1:00PM ET
* 11/20-11/25 at 9PM ET
* and once a day at 11:59PM ET until 12/02

We will update these scores on Gradescope after running:
* Guesser: precision and recall,
* Buzzer: expected win, best_score, buzz ratio, buzz position.

Extra Credit
======

* [Up to 10 Points] Improve the performance of the overall system.  We already
  talked about the Guesser, but for the Buzzer there are vey easy ways to do
  this: we are forming the `text` field of the examples in a fairly naive way.
  We could add more information or format it better.  A more involved (but
  likely better) is to further extend the model to better encode additional
  floating point features (like you did in the feature engineering homework).

* [Up to 5 Points] Add additional sources of information to DSPy retrieval
  (e.g., Wikipedia).

* [Up to 5 Points] Explicitly add multihop reasoning to DSPy guesser to solve
  subproblems.

* [Up to 5 Points] Experiment with what layers are most necessary for the best
  improvements and test values of alpha and rank that work best (you cannot
  use tiny datasets for this, unfortunately, so this requires a GPU, probably
  ... not a great big one, as any GPU will likely be fine).  Make sure in
  addition to any accuracy / buzz ratio numbers you provide you also count the
  number of parameters.

* [Up to 3 Points] The training code in `train` are taken directly from the
  Huggingface examples and I didn't think too much about them.  It's not clear
  that they're a good fit for the data.  Can you find something substantially
  better?  (Keeping the model / adaptation / etc. constant.)  For example, we
  know accuracy isn't exactly what we want, but that's what's being optimized.

Hints
=================

If you have access to a GPU, you should use it whenever you can for both
Ollama and for the BERT adaptation.

You'll want to tune your tf-idf retriever. 

    ./venv/bin/python3 eval.py --evaluate guesser --questions ../data/qanta.guessdev.json.gz --cutoff 200 --num_guesses 10

Some ideas:
     * Add bigrams to the vocabulary
     
     * Make sure your eval is not on the full question (the default).  A
       suggestion would be to do 200 characters (as above), but less is
       obviously better.  Recall is more important than precision.

     * Once it's settled, you'll want to start tuning your Ollama Guesser

You can reuse the eval script for that:

    ./venv/bin/python3 eval.py --evaluate guesser --questions ../data/qanta.buzztrain.json.gz --limit 100 --num_guesses 1 --guesser_type Ollama

    * It should be better than tf-idf ... otherwise, what's the point?

    * I used buzztrain this time because it's used guessdev already.  It's
      okay to use a little bit of buzztrain on this to see how well it's
      doing.

There are also options for improving the Buzzer:

      * You can use a different base model

      * You can add more fields in the `dataset_from_questions` function

      * You can use a different objective function

Once that's trained, you can evaluate it as usual:

     ./venv/bin/python3 eval.py --evaluate buzzer --questions ../data/qanta.buzzdev.json.gz --limit 100 --num_guesses 1 --buzzer_guessers=Tfidf --buzzer_type='lorabert'

It could be that you've done a really good job getting DSPy to do everything
you've need (or you heavily used buzzer train / dev).  In which case you can
use threshold_buzzer.py if the BERT training is becoming difficult.
      
      ./venv/bin/python3 -i eval.py --evaluate buzzer --questions ../data/qanta.buzzdev.json.gz --limit 100 --num_guesses 1 --buzzer_guessers=Tfidf --buzzer_type='threshold' --threshold_buzzer_threshold=0.4

If you do this, make sure you set `parameters.py` to use that buzzer by
default (and not LoRABERT).

**WARNING**: It's very easy to get things *running* by making minimal
  modifications to the provided code.  Getting things working well will
  require real time and preparation.

FAQ
========

*Q: Why do the unit tests use "encoder" but the template code uses "transformer"?*

*A:* The unit tests are using a "real" (but very tiny) BERT, while the template code is using DistilBERT (but much larger).  They are packaged slightly differently.

*Q: What Ollama models can I use?*

*A:* You may use: Gemma3:4b, Qwen3:4b and Llama3.2:3b   [Updated 20 November 2025]

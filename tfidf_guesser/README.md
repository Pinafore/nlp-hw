Nearest Neighbor Question Answering (Guessing)
=

Overview
--------

In this homework you'll implement a nearest neighbor approach to answering
questions.  Given a question you find the most similar question to it, and
answer with the answer to that question.

This first step is meant to be very simple.  If you're spending a
lot of time on this assignment, you are either:

* seriously over-thinking the assignment
* trying to reimplement too much of the assignment

Most of this assignment will be done by calling libraries that have already
been implemented for you.  If you are over-implementing, you are generating
extra work for yourself and making yourself vulnerable to errors.

That said, the second part of the homework---doing as well as you can
buzzing---is meant to be more fun and open-ended.

You'll turn in your code on Gradescope.

What you have to do
----

Coding (15 points in the tfidf_guesser.py):

1.  (Optional) Store necessary data in the constructor so you can do retrieval later.
1.  Modify the _train_ function so that the class stores what it needs to store to guess at what the answer is.
1.  Modify the _call_ function so that it finds the closest indicies (in terms of *cosine* similarity) to the query.

Analysis (5 points):

1.  What answers get confused with each other most easily?  What kinds of
    mistakes does this guesser make?
1.  How does this guesser compare to GPT?  (Remember that the cached guesser from the feature engineering homework came from GPT3, so you could either use your old code or adapt with multiple guessers here!)
1.  Compute recall as you increase the number of guesses (i.e. `max_n_guesses`).

Accuracy (10 points): How well you do on the recall leaderboard.

What you don't have to do
-------

You don't have to (and shouldn't!) compute tf-idf yourself.  We did that in
a previous homework, so you can leave that to the professionals.  We encourage
you to use the tf-idf vectorizer from sklearn: play around with different settings of the
paramters.  You probably shouldn't modify it, but it's probably useful to
understand it for future homeworks (you'll need to write/call code like it in
the future).

You also don't have to save the vectorizer and tfidf representations, we'll do
that for you.  Take a look at the base guesser class, it needs to track the
questions and answer.  This will also be saved to a pickle.

Unit tests
-

To test that your basic functionality works, run the provided unit
tests:

     $ python3 guesser_test.py
                                          This capital of England                          Maine                          Maine 0.545
                                                                                                                         Boston 0.528
                                The author of Pride and Prejudice                    Jane_Austen                    Jane_Austen 0.913
                                                                                                                    Jane_Austen 0.779
                                  The composer of the Magic Flute        Wolfgang_Amadeus_Mozart        Wolfgang_Amadeus_Mozart 0.691
                                                                                                        Wolfgang_Amadeus_Mozart 0.653
           The economic law that says 'good money drives out bad'                  Gresham's_law                  Gresham's_law 0.751
                                                                                                                  Gresham's_law 0.721
     located outside Boston, the oldest University in the United       College_of_William_&_Mary      College_of_William_&_Mary 0.450
                                                                                                              Rhode_Island 0.426

Your numeric results might not exactly match the similarities here,
but the ranking should still be consistent.

You might notice the batch guess test failed with the current `batch_guess` function. You could either implement the function (though not required) or use the same function from its parent class.

What to turn in
-
For the guesser submission:
1. Submit your _tfidf_guesser.py_ file (so we could retrain your model!)
2. Submit your _analysis.pdf_ file (no more than one page; pictures
    are better than text)

For the extra credit submission:
1. Submit your _tfidf_guesser.py_ file
2. Submit the ``TfidfGuesser.answers.pkl``,
   ``TfidfGuesser.questions.pkl``, ``TfidfGuesser.tfidf.pkl`` and the
   ``TfidfGuesser.vectorizer.pkl`` files that encode your model. 
3. For the extra credit buzzer, also upload your _params.py_ and _features.py_ files and your classifier pickles.
    

Extra Credit
=

There will be two different places to submit your code on Gradescope:
one that only tests the guesser, one that specifically tests the
buzzer.  The guesser evaluation will retrain your model, the buzzer
evaluation will use the the uploading model directly.  
1. Optimize the retrieval mechanism by tuning parameters, weighting, and/or using
   different tokenizers/vocabularies.
2. Do well in the overall leaderboard (while overall buzz ratio and accuracy is important, more
   important is using features that take advantage of tfidf guesser features or
   multiple guessers.)
3. Add additional tf-idf guessers (e.g., from the provided Wikipedia pages).  You can create an additional
    guesser if you want to keep it separate from the tfidf_guesser.  If you do
    that, make sure to upload that file too.
5. You can and should use multiple guessers (e.g., it's allowed to use
   the GPT and tf-idf guesser).  You can also create a new guesser.

What makes this more fun than the last feature engineering assignment is that you have full control over the buzzer now, and you get to change what it's producing.  So now you can do more than create features *given* the guesses, you can now fix the guesser's problems as well!

Example
-

Let's first test out the train function; you must run this before the eval
function, because this establishes your tf-idf index. (You might want to run `mkdir -p models` first!)

    python3 guesser.py --guesser_type=Tfidf \
    --question_source=gzjson \
    --questions=../data/qanta.guesstrain.json.gz \
    --logging_file=guesser.log \
    --limit=10 
    Setting up logging
    INFO:root:Using device 'cpu' (cuda flag=False)
    INFO:root:Initializing guesser of type Tfidf
    INFO:root:Loading questions from ../data/qanta.guesstrain.json.gz
    INFO:root:Read 10 questions
    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 2473.93it/s]
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 590747.04it/s]
    INFO:root:Trained with 57 questions and 57 answers filtered from 10 examples
    INFO:root:Creating tf-idf dataframe with 57


This outputs the vectorizer (which turns text into a matrix) into the models
directory (you might need to create the directory called ``models``).  After
you've done that, you can now run the guesser.

*BEWARE*: The code I've given you runs end to end, but it's not correct.  It
 creates random representations of all of the documents and the retrieval
 component always returns the zeroth document.  This is why eval will always
 answer the same darn answer to all of the questions.  You'll obviously need
 to fix this.

This is an example of what your code output should look like when you
evaluate how well the recall is working:
```
> python3 eval.py --evaluate=guesser --question_source=gzjson \
--questions=../data/qanta.guessdev.json.gz --limit=500

hit 0.20
===================

               guess: William_Cullen_Bryant
              answer: William_Cullen_Bryant
                  id: 102410
                text: A poem by this man that notes that "Soft airs, and song, and light,
                      and bloom" should keep friends "lingering by my tomb" was praised by
                      Edgar Allan Poe in "The Poetic Principle". Another of his poems
                      contains the line "Truth, crushed to earth, shall rise again", which
                      was quoted by MLK Jr. in his "Give Us the Ballot" speech. This author
                      of "June" and "Battle-field" gained early popularity with a political
                      satire that opens with the couplet "When private faith and public
                      trust are sold, and traitors barter liberty for gold". He wrote a poem
                      that describes a "a Power whose care / teaches thy way along that
                      pathless coast" and asks its addressee if it "seek'st" the "plashy
                      brink". That poem by this man notes that "the abyss of heaven / Hath
                      swallowed u
--------------------
               guess: Ophelia
              answer: Ophelia
                  id: 141393
                text: This character admonishes another to not show others the "the steep
                      and thorny way to heaven" while treading the "primrose path of
                      dalliance." In one scene, this character hands out flowers and herbs
                      but keeps rue to herself. A character picks up a skull at this
                      character's burial, at which this character's brother jumps into the
                      grave. This
--------------------
               guess: Leigh_Hunt
              answer: Leigh_Hunt
                  id: 102151
                text: This poet imagines "the strongest thread / Of our frail plant" saying
                      "Patience and Gentleness is Power" in "On a Lock of Milton's Hair."
                      John Keats wrote that this man
--------------------
               guess: Circular_dichroism
              answer: Circular_dichroism
                  id: 102305
                text: Software packages such as CDSSTR, SELCON, and CONTIN can be used to
                      analyze spectra produced by this technique. The Chirascan is a brand
                      of spectrometer used for this technique. Spectrometers for this
                      technique can be calibrated using pantolactone or 10-camphorsulfonic
                      acid as standards. Calc
--------------------
               guess: Piano_sonata
              answer: Piano_sonata
                  id: 93242
                text: Haydn's final numbered composition of this type opens with rolled
                      chords in a dotted rhythm followed by descending passages in double
                      thirds. That piece of this type features a second movement
                      surprisingly in E major after a first movement in E-flat major. Pieces
                      of this type comprise the bulk of the output of Muzio Clementi, who
                      wrote over a hundred and established the genre in the Classical era.
                      Mozart's compositions of this type include his K.545, which is his
                      sixteenth in C major, labeled "semplice" or "for beginners," an
--------------------
               guess: The_God_of_Small_Things
              answer: The_God_of_Small_Things
                  id: 102399
                text: One character in this novel used to practice making sad faces in the
                      mirror while quoting Sydney Carton's lines from A Tale of Two Cities.
                      After falling in love with the Irish Father Mulligan and joining a
                      convent, another character in this novel obtains a diploma in
                      ornamental gardening; that character is later humiliated when a group
                      of protesters force her to wave a red flag and chant Communist
                      slogans. A character in this novel thinks "Two Thoughts", namely that
                      "Anything can happen to anyone" and "It's best to be prepared", while
                      working at Paradise Pickles. During a performance of the Sound of
                      Music in this novel, one of the protagonists
--------------------
               guess: To_a_Skylark
              answer: To_a_Skylark
                  id: 93292
                text: This poem's speaker questions the "joy we ever should come near" if
                      "we were things born / Not to shed a tear" and asks, "What objects are
                      the fountains / Of thy happy strain?" Images in this poem include a
                      "rose embowered / In its own green leaves" and a "glow-worm golden /
                      In a dell of dew." This poem's speaker states "The world should listen
                      then Â– as I am listening now" after declaring that if the title
                      figure would teach him "half the gladness / That thy brain must know,
                      / Such harmonious madness / From my lips would flow." This poem begins
                      with its speaker exclaiming, "Hail to thee, blithe Spirit!" For 10
                      points, name this poem addressing a bird, by Percy Shel
--------------------
               guess: Viola
              answer: Viola
                  id: 93464
                text: Ernest Bloch's Suite Hebraique is for this solo instrument and
                      orchestra, and Max Bruch wrote a double concerto for clarinet and this
                      instrument. A rhapsody for this instrument and orchestra was wr
--------------------
               guess: Cain
              answer: Cain
                  id: 141436
                text: According to Talmudic tradition, this figure was accidentally killed
                      by his great-grandson Lamech. In the Quran, this man wishes to marry
                      the beautiful Aclima. This man fathers several sons, including Enoch,
                      while wandering in the Land of Nod, but all of his descendents are
                      later killed in the Flood. This farmer and brother of Seth g
--------------------
               guess: North_Macedonia
              answer: North_Macedonia
                  id: 93194
                text: This country's Titov Veles district is known for its high quality
                      opium. A campaign to build nationalist monuments in this country is
                      known as antiquisation. The Golem Grad, home to ruined churches and
                      thousands of snakes, can be found in this country's majority portion
                      of Lake Prespa. Using Motorola's Canopy technology, this country was
                      the first to
--------------------
=================
miss 0.69
===================

               guess: Seven
              answer: Symphony_No._2_(Mahler)
                  id: 102021
                text: The first movement of Lutoslawski's symphony of this number consists
                      of seven
--------------------
               guess: Aldehyde
              answer: Sugars
                  id: 102058
                text: One type of these compounds can be detected via the reduction of
                      copper acetate to copper oxide in Barfoed's test, while two different
                      types of these compounds can be distinguished using a solution of
                      resorcinol and hydrochloric acid in Seliwanoff's test. These compounds
                      isomerize through an enediol intermediate via the Lobry de Bruyn-van
                      Ekenstein reaction. These compounds can be shortened by one carbon by
                      oxidizing them with bromine water, then a
--------------------
               guess: Peasants'_Revolt
              answer: Jōan
                  id: 93487
                text: After stopping a woman with this name on Blackheath, Wat Tyler
                      supposedly kissed her hand and attached an escort to her company. This
                      was the na
--------------------
               guess: Glycerol
              answer: Ethylene_glycol
                  id: 102353
                text: In toxicity workups, the level of this molecule can be estimated by
                      multiplying the osmolar gap by a factor of 6.2. This molecule is the
                      larger of the two compounds whose poisoning is counteracted by
                      4-methylpyrazole, also known as fomepizol. Thiamine and pyridoxine can
                      inhibit the production of this molecule's metabolite oxalic acid.
                      Oxirane is the usual precursor of this molecule in synthetic pathways,
                      such as the Shell-developed OMEGA process. Under acid catalysis,
                      aldehydes react with this molecule to form 1,3-dioxolanes, which is a
                      common strategy for protecting carbonyls using acetals. This compound
                      is reacted with terephthalic acid to produce the thermoplastic polymer
                      PET. In its most common usage, this molecule is often eschewed in
                      favor of its safer rel
--------------------
               guess: Book_of_Mormon
              answer: The_World_as_Will_and_Representation
                  id: 93181
                text: This book rej
--------------------
               guess: Slobodan_Milošević
              answer: Khair_ad-Din
                  id: 102427
                text: At the request of his master, this man tried to kidnap the beauty
                      Giulia Gonzaga - after he failed, he sacked the town of Sperlonga, and
                      Giulia had a knight executed for seeing her naked. The force of this
                      non-monarch allegedly gave up on a siege after the obese laundress
                      Catherine Segurane mooned them. This man ordered that church bells not
                      be rung because they disturbed his sleep while wintering his troops in
                      Toulon, and he was assisted by a man called "the Great Jew," Sinan
                      Reis. He met with Antonio Rincon to negotiate an alliance between his
                      monarch and Francis I, but his force was defeated at the Battle of
                      Preveza by an allied fleet assembled by Pope Paul III in 1538. Both
                      this man and his elder br
--------------------
               guess: Pablo_Neruda
              answer: Anne_Bradstreet
                  id: 141454
                text: One of the two title figures of a poem by this author asks "must my
                      self dissect my tatter'd state, /  Which Amazed Christendom stands
                      wondering at?" This author wondered "What gripes of wind my infancy
                      did pain, / What tortures I in breeding teeth sustain?" in one of four
                      "quaternion" poems. This author of "Dialogue Between Old Eng
--------------------
               guess: Raven
              answer: Crows
                  id: 102367
                text: The Basque goddess Mari turned a shepherd who built a house too close
                      to one of her caves into this animal. One of these animals named Kutkh
                      turns into an old man after regurgitating the Earth in a Chukchi myth.
                      The protector deity Gonpo Jarodonchen is r
--------------------
               guess: Plato
              answer: Trobriand_people
                  id: 93348
                text: To get married in this culture, a man and woman sit in front of his
                      house in the morning until the bride's mother acknow
--------------------
               guess: Contract_with_America
              answer: None
                  id: 102292
                text: This hypothetical act titles a book by Ramona Naddaff, which argues
                      that it is actually an act of cultural and aesthetic discussion. Iris
                      Murdoch's The Fire and the Sun defends this actionon the grounds that
                      it is easier to copy a bad man than a good man. The so-called
                      "psychological critique" defends this act by citing how art nourishes
                      the irrational part of the soul. The "metaphysical critique" which
                      defends this act uses the example of God creating the Form of
                      furniture, a carpenter creating furniture, and finally, a painter
                      making a representation of a bed. This suggested action seems to
                      contradict itself, since it is immediately followed by the most
                      imaginative portion of the dialogue, the story of the Myth of Er. For
                      10 points, name this act that occurs at the beginning of the tenth
                      book of Plato's Republic, in which Socrat
--------------------
=================
close 0.12
===================

               guess: Diffraction
              answer: Solow–Swan_model
                  id: 102207
                text: A "balanced path" of a certain quantity in this formalism is achieved
                      with assumptions that ensure its predictions correspond to Nicholas
                      Kaldor's stylized facts. When this formalism's central equation
                      satisfies Euler's homogeneity theorem, there is no need to specify the
                      ownership of firms since their profit in the long run is zero. Mankiw,
                      Romer, and Weil proposed augmenting this formalism with a distinction
                      between human and physical capital. An extension of this formalism
                      that adds the assumptions of Walrasian equilibrium and endogenous
                      household savings is named for Ramsey, Cass, and Koopmans. Equilibrium
                      in this formalism is achieved when g
--------------------
               guess: New_Orleans
              answer: San_Francisco
                  id: 141320
                text: A street in this city that divided its poor and rich sections was
                      nicknamed "the Slot." Amadeo Giannini's Bank of Italy, which grew into
                      the Bank of America, was founded in this city. Mayor Eugene Schmitz,
                      the puppet of Union Labor Party boss Abe Ruef, created the Committee
                      of Fifty to respond to a disaster in this city. This city exhibited
                      its recovery from that disaster by hosting the extravagant 1915
                      Panama-Pacific International Exposition. A fire in this city destroyed
                      many opulent Nob Hill mansions. Around 80% of this city
--------------------
               guess: Tang_dynasty
              answer: Abbasid_Caliphate
                  id: 93307
                text: A "beveled" style of decorative carving emerged from this dynasty, as
                      did a code of literary etiquette called adab. This dynasty oversaw the
                      failure of its irrigation system in the "black land," or Sawab, south
                      of its capital. Three brothers serving this dynasty sketched many
                      hydraulic automata in a Book of Ingenious Devices. This dynasty
                      briefly persecuted heretics at its mihna court. This dynasty's capital
                      had four gates spaced 90 degrees apart around perfectly round walls.
                      The Barmak
--------------------
               guess: Amarna
              answer: Maya_Hero_Twins
                  id: 93359
                text: In one story, these figures send some ants to cut down flowers and
                      bring them back. Scholars debate why a member of this group is male
                      but has a name beginning with a feminine pref
--------------------
               guess: Phase_diagram
              answer: Turing_machine
                  id: 93494
                text: Langston's ant and turmites are two-dimensional analogues of these
                      constructs. Recursively enumerable languages are equivalent to the
                      class of languages recognized by these constructs. The universal
                      variant of these constructs can simulate every possible one of these
                      constructs. Attaching one of these to an "oracle" can make it
                      stronger, and a busy beaver is a specific variant of these constructs
                      that executes the maximum number of steps. Lambda calculus is
                      equivalent in power to these constructs according to a hypothesis
                      named after Alonzo Church and these devices' namesake. For 10 points,
                      name these theo
--------------------
               guess: Freyr
              answer: Odin
                  id: 102356
                text: According to Snorri, a king of Uppsala named Swegde spent many years
                      trying to enter the realm of this god. A euhemerized account of this
                      figure's rule states that, after this figure died at Swithiod, he went
                      to Godheim. According to the Heimskringla, this man was the second
                      husband of Skadi. H
--------------------
               guess: The_Social_Contract
              answer: Shinto
                  id: 141359
                text: A festival in this religion includes the giving of "thousand year
                      candy," and a construction site sacred to this religion has its ground
                      broken using a sickle, hoe, and spade. A practice in this religion
                      involves writing prayers or wishes on small wooden plaques. This
                      religion's emphasis on purification is why its places of worship all
                      have a bas
--------------------
               guess: Max_Weber
              answer: Mormonism
                  id: 141406
                text: This religion holds that salvation is denied only to the "sons of
                      perdition" who have perfect knowledge of the truth but willingly turn
                      away from it; everyone else, according to this religion, will be
                      assigned to one of the celestial, telestial, or terrestrial "kingdoms
                      of glory". Men who follow this religion may choose to join the
                      Melchizedek priesthood, and every worthy male is considered a member
                      of this religion's Aaronic priesthood. This religion is named after
                      the fourth-century prophet who allegedly engraved an account of his
                      revela
--------------------
               guess: Triple_bond
              answer: Peroxide
                  id: 102295
                text: The terpene ascaridole unusually possesses this functional group in a
                      bridging position, while three bridges containing this functional
                      group link two tertiary amines in HMTD. The smallest compound with
                      this functional group is produced in the anthraquinone process. A Hock
                      rearrangement of this functional group attached to cumene is used to
                      produce phenol and acetone in the cumene process. These compounds are
                      used to oxidize sulfides to sulfoxides or sulfones, and they are
                      produced in autoxidation reactions. Oxidation of ketones to esters via
                      the Baeyer-Villiger reaction and non-stereospecific epoxidation of
                      alkenes are typically catalyzed with a compound containing a hybrid of
                      this functional group and a carboxylic acid. mCPBA contains this
                      functional group, which is characterized by an oxygen-oxygen single
                      bond. For 10 points, name this functional group found in
--------------------
               guess: Dmitry_Kabalevsky
              answer: Romain_Rolland
                  id: 102161
                text: One of this author's characters flees to Switzerland after he kills a
                      soldier at a bloody May Day celebration thrown by some syndicalists
                      during which his friend Olivier dies. That title character created by
                      him flees to Paris after killing yet another soldier, who was trying
                      to rape the peasant girl Lorchen. This man's study of eastern
                      mysticism, which includes a three-volume life of Swami Vivekananda,
                      inspired Sigmund Freud's discussion of the oceanic feeling in
                      Civilization and its Discontents. Dmitri Kabalevsky created an
                      operatic adaptation of this man's novel Colas Breugnon. Because it
                      "flows like a river," he dubbed his longest work, which includes
                      1911's The Burning Bush and 1904's Dawn, a "roman-fleuve". That novel
                      opens by describing a son of Melchior Krafft
--------------------
=================
```

This is different from the guesser last time because we now have
multiple guesses at once.  In other words, rather than only guessing
"London" given the question "Name this UK capital", you'll get more
guesses like "Cardiff" and "Edinburgh".  So this means that you'll be
evaluating more guesses, and you may want to incorporate more features
to distinguish middling guesses from the top guesses.

Good Enough
-

For the "Good Enough" threshold, you need to implement tfidf on par
with the baseline.  You do not need to do additional feature
engineering.  

Extra Credit
-

We have a separate leaderboard for the extra credit.  Now that you
have full control over the Guesser(s), you can perhaps be even more
creative than you were on the last assignments with your features.  

One thing you'll notice is that because there's more than one guess,
you can and should be using the other guesses to help you do a better
job of crafting features:

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        raise NotImplementedError(
            "Subclasses of Feature must implement this function")

The ``other_guesses`` structure is a dictionary, where the keys are
the names of the guesser and then you have all of the guesses that
they produced.

    {
      "Tfidf": [
        {
          "guess": "",
          "confidence": 0.37647900686306596,
          "question": "This equation's spherical analog uses a great (*) circle as the geodesic."
        },
        {
          "guess": "Gradient_descent",
          "confidence": 0.31423377416353393,
          "question": "is a lock-free implementation of an asynchronous version of this algorithm, while Nesterov and Rumelhart et al."
        },
        {
          "guess": "Kerr_effect",
          "confidence": 0.2772825999628262,
          "question": "The collapse and revival of a quantum state via this effect was observed by Schoelkopf et al, and Degert et al induced it via terahertz cycling."
        },
        {
          "guess": "",
          "confidence": 0.2541601924817103,
          "question": "His daughter Edie was at one time married to Geraldo Rivera."
        },
        {
          "guess": "Kurt_Vonnegut",
          "confidence": 0.2541601924817103,
          "question": "His daughter Edie was at one time married to Geraldo Rivera."
        },
        {
          "guess": "Möbius_aromaticity",
          "confidence": 0.24926279107473492,
          "question": "The first synthesis of a molecule with this property was performed by Herges et al."
        },
        {
          "guess": "Travelling_salesman_problem",
          "confidence": 0.24657150329865837,
          "question": "A 3/2 approximative algorithm to solve it is the Christofides algorithm."
        },
        {
          "guess": "Frida_Kahlo",
          "confidence": 0.243589161276024,
          "question": "For 10 points, name this unibrowed wife of Diego Rivera."
        },
        {
          "guess": "SEARCH",
          "confidence": 0.24244180585678313,
          "question": "One algorithm for it runs in log N time by splitting the data in half at each step; that is the (*) binary algorithm for doing this."
        },
        {
          "guess": "Torus",
          "confidence": 0.24171360134763703,
          "question": "They can be created by revolving a circle around an axis coplanar to the circle's diameter."
        },
        {
          "guess": "Ribosome",
          "confidence": 0.23915538398196445,
          "question": "The first atomic structure of one of these was published by Ban et al."
        },
        {
          "guess": "Integer_factorization",
          "confidence": 0.238308007274263,
          "question": "Amethod of doing this which finds a cycle in a pseudo-random sequence is Pollard's rho algorithm.Shor developed a polynomial-time algorithm for doing this on a (*) quantum computer."
        },
        {
          "guess": "",
          "confidence": 0.23762897194751045,
          "question": "Performers of this action might encounter a shalshelet in one of four locations, but more commonly see a pazer or et-nachta while doing it."
        },
        {
          "guess": "Garbage_collection_(computer_science)",
          "confidence": 0.2339602358183885,
          "question": "The Deutsch-Schorr-Waite algorithm is an example of a pointer-reversal algorithm for doing it."
        },
        {
          "guess": "Pseudorandom_number_generation",
          "confidence": 0.23292464419908454,
          "question": "Some common ways of doing this utilize Schrage's multiplication algorithm for increased performance, and a fast method of doing this was developed by Matsumoto and Nishimura."
        },
        {
          "guess": "Scheduling",
          "confidence": 0.232168804860847,
          "question": "An optimal algorithm for performing this task which minimizes the average waiting time is the shortest-job-first algorithm, which is itself a special case of a priority algorithm for doing this."
        },
        {
          "guess": "",
          "confidence": 0.230347158472583,
          "question": "Bakun and A.G. Lindh's Parkfield experiment was a failed attempt at doing this task, which is the goal of the the M8 algorithm."
        },
        {
          "guess": "Cluster_analysis",
          "confidence": 0.23005424325379417,
          "question": "The 2014 KDD conference gave a test of time award to a 1996 paper by Kriegel et al that proposed an algorithm for accomplishing this task which improved upon the CLARANS algorithm to work against data sets of arbitrary shapes."
        },
        {
          "guess": "Parsing",
          "confidence": 0.22902004059465228,
          "question": "Earley's algorithm and the CYK algorithm perform this task."
        },
        {
          "guess": "Online_algorithm",
          "confidence": 0.228495591909524,
          "question": "Welford's algorithm for variance is this type of algorithm."
        },
        {
          "guess": "Augusto_Pinochet",
          "confidence": 0.22793925550016003,
          "question": "This leader was also opposed by the group FPMR, a patriotic front named for Manuel Rodriguez."
        },
        {
          "guess": "Iridium",
          "confidence": 0.22766311884032236,
          "question": "A photo-activate-able piano stool compound containing this metal was characterized by both Graham et al and Bergman et al and was used to catalyze C-H bond activation."
        },
        {
          "guess": "Multiplication",
          "confidence": 0.2266223259445582,
          "question": "Until its replacement by the Furer algorithm, the Schonhage-Strassen algorithm was the asymptotically fastest way of doing this to large numbers."
        },
        {
          "guess": "Ghrelin",
          "confidence": 0.22467213803125147,
          "question": "Produced by P/D1 cells, it was first discovered by Kojima et al."
        }
      ]
    }

Remember that if you want to inspect what the features look like, you
can always use the ``features.py`` script, which generates the
training data you had for the logistic regression homework:

    python3 features.py --guesser_type=Tfidf --limit=100  --question_source=gzjson --TfidfGuesser_filename=models/TfidfGuesser  --questions=../data/qanta.buzztrain.json.gz --buzzer_guessers=Tfidf --json_guess_output=temp.out

If you look at the outputs, you can see how multiple guesses might be
useful.  The ``consensus`` features counts up all the times that a
guess has been made (higher is better, presumably).

    {"guess:Garbage_collection_(computer_science)": 1, "Tfidf_confidence": 0.24944632130872874, "consensus": 6, "Length_char": 0.3333333333333333, "Length_word": 0.18666666666666668, "Length_ftp": 0, "Length_guess": 3.6375861597263857, "Frequency_guess": 2.1972245773362196, "Category_category:Science": 1, "Category_year": 3.4011973816621555, "Category_subcategory:Science Computer Science": 1, "Category_tournament:ACF Winter": 1, "label": true}
    {"guess:Garbage_collection_(computer_science)": 1, "Tfidf_confidence": 0.3401856393594229, "consensus": 7, "Length_char": 0.5577777777777778, "Length_word": 0.4, "Length_ftp": 1, "Length_guess": 3.6375861597263857, "Frequency_guess": 2.1972245773362196, "Category_category:Science": 1, "Category_year": 3.4011973816621555, "Category_subcategory:Science Computer Science": 1, "Category_tournament:ACF Winter": 1, "label": true}
    {"guess:Garbage_collection_(computer_science)": 1, "Tfidf_confidence": 0.3253436462425905, "consensus": 7, "Length_char": 0.7266666666666667, "Length_word": 0.5733333333333334, "Length_ftp": 1, "Length_guess": 3.6375861597263857, "Frequency_guess": 2.1972245773362196, "Category_category:Science": 1, "Category_year": 3.4011973816621555, "Category_subcategory:Science Computer Science": 1, "Category_tournament:ACF Winter": 1, "label": true}
    {"guess: ": 1, "Tfidf_confidence": 0.32610226495081074, "consensus": 0, "Length_char": -0.7666666666666667, "Length_word": -0.72, "Length_ftp": 0, "Length_guess": 0.6931471805599453, "Frequency_guess": 9.103089181229207, "Category_category:Fine Arts": 1, "Category_year": 3.4011973816621555, "Category_subcategory:Fine Arts Visual": 1, "Category_tournament:ACF Winter": 1, "label": false}

This becomes more relevant for using multiple guessers.  If we use
both guessers with the Gpr guesser as the primary guesser, we can now
see how this can help us.  So we generate the features (use a similar
command line for this to be your buzzer).

    python3 features.py --limit=100  --question_source=gzjson --TfidfGuesser_filename=models/TfidfGuesser  --questions=../data/qanta.buzztrain.json.gz --buzzer_guessers Tfidf Gpr --primary_guesser Gpr --json_guess_output=temp.out

Now we can see for this question: 

    {"guess:William Carlos Williams": 1, "Tfidf_confidence": 0.3720513912171859, "Gpr_confidence": -0.34099804599433337, "consensus_count": 0, "consensus_match": 0, "Length_char": -0.7755555555555556, "Length_word": -0.7466666666666667, "Length_ftp": 0, "Length_guess": 3.1780538303479458, "Frequency_guess": 3.4011973816621555, "Category_category:Literature": 1, "Category_year": 3.4011973816621555, "Category_subcategory:Literature American": 1, "Category_tournament:ACF Winter": 1, "label": false}
    {"guess:Ishmael": 1, "Tfidf_confidence": 0.37931701332166823, "Gpr_confidence": -0.35492457459849996, "consensus_count": 0, "consensus_match": 0, "Length_char": -0.5511111111111111, "Length_word": -0.49333333333333335, "Length_ftp": 0, "Length_guess": 2.0794415416798357, "Frequency_guess": 2.0794415416798357, "Category_category:Literature": 1, "Category_year": 3.4011973816621555, "Category_subcategory:Literature American": 1, "Category_tournament:ACF Winter": 1, "label": false}
    {"guess:Moby-Dick": 1, "Tfidf_confidence": 0.28765034540766443, "Gpr_confidence": -0.21047059701, "consensus_count": 1, "consensus_match": 0, "Length_char": -0.3333333333333333, "Length_word": -0.22666666666666666, "Length_ftp": 0, "Length_guess": 2.302585092994046, "Frequency_guess": 3.7376696182833684, "Category_category:Literature": 1, "Category_year": 3.4011973816621555, "Category_subcategory:Literature American": 1, "Category_tournament:ACF Winter": 1, "label": false}
    {"guess:Queequeg": 1, "Tfidf_confidence": 0.27029424160595916, "Gpr_confidence": -0.0862097396453, "consensus_count": 2, "consensus_match": 0, "Length_char": -0.1111111111111111, "Length_word": 0.013333333333333334, "Length_ftp": 0, "Length_guess": 2.1972245773362196, "Frequency_guess": 1.0986122886681098, "Category_category:Literature": 1, "Category_year": 3.4011973816621555, "Category_subcategory:Literature American": 1, "Category_tournament:ACF Winter": 1, "label": true}
    {"guess:Queequeg": 1, "Tfidf_confidence": 0.24726462021560894, "Gpr_confidence": -0.047181845223625, "consensus_count": 2, "consensus_match": 0, "Length_char": 0.13111111111111112, "Length_word": 0.26666666666666666, "Length_ftp": 0, "Length_guess": 2.1972245773362196, "Frequency_guess": 1.0986122886681098, "Category_category:Literature": 1, "Category_year": 3.4011973816621555, "Category_subcategory:Literature American": 1, "Category_tournament:ACF Winter": 1, "label": true}
    {"guess:Queequeg": 1, "Tfidf_confidence": 0.23335231906119935, "Gpr_confidence": -0.01780669447, "consensus_count": 2, "consensus_match": 1, "Length_char": 0.3377777777777778, "Length_word": 0.48, "Length_ftp": 0, "Length_guess": 2.1972245773362196, "Frequency_guess": 1.0986122886681098, "Category_category:Literature": 1, "Category_year": 3.4011973816621555, "Category_subcategory:Literature American": 1, "Category_tournament:ACF Winter": 1, "label": true}
    {"guess:Queequeg": 1, "Tfidf_confidence": 0.21786668240455015, "Gpr_confidence": -0.0030035892337500003, "consensus_count": 2, "consensus_match": 1, "Length_char": 0.5555555555555556, "Length_word": 0.7333333333333333, "Length_ftp": 0, "Length_guess": 2.1972245773362196, "Frequency_guess": 1.0986122886681098, "Category_category:Literature": 1, "Category_year": 3.4011973816621555, "Category_subcategory:Literature American": 1, "Category_tournament:ACF Winter": 1, "label": true}
    {"guess:Queequeg": 1, "Tfidf_confidence": 0.23749024257054557, "Gpr_confidence": -0.00091445903425, "consensus_count": 4, "consensus_match": 1, "Length_char": 0.74, "Length_word": 0.9066666666666666, "Length_ftp": 1, "Length_guess": 2.1972245773362196, "Frequency_guess": 1.0986122886681098, "Category_category:Literature": 1, "Category_year": 3.4011973816621555, "Category_subcategory:Literature American": 1, "Category_tournament:ACF Winter": 1, "label": true}

As we go deeper into the question, the tf-idf search is turningg up
more matches, so the consensus count is going up.  This is a great
feature that can help the `Gpr_confidence` actually going down.  

Speaking of, you might want to play around how that confidence is
computed as well.  Take a look at the cache object (use `buzzdev` below as an example):

    zless ../models/buzzdev_gpr_cache.tar.gz
      "After this character relates a story about how he didn't know the proper way to use a wheelbarrow, he": {
        "guess": "William Carlos Williams",
        "confidence": [
          [
            "William",
            -1.0181785
          ],
          [
            " Carlos",
            -0.004757869
          ],
          [
            " Williams",
            -5.7768983e-05
          ]
        ]
      },
      "After this character relates a story about how he didn't know the proper way to use a wheelbarrow, he tells of how a captain dining with his father mistakenly rubbed his hands in a punch bowl. This \"sea": {
        "guess": "Ishmael",
        "confidence": [
          [
            "I",
            -1.00852
          ],
          [
            "sh",
            -0.41000807
          ],
          [
            "ma",
            -0.001125095
          ],
          [
            "el",
            -4.5133394e-05
          ]
        ]
      },
      "After this character relates a story about how he didn't know the proper way to use a wheelbarrow, he tells of how a captain dining with his father mistakenly rubbed his hands in a punch bowl. This \"sea Prince of Wales\" leaves his home by hiding out in a canoe near a coral reef, and he is mistakenly": {
        "guess": "Moby-Dick",
        "confidence": [
          [
            "M",
            -0.8082461
          ],
          [
            "oby",
            -0.027521435
          ],
          [
            "-D",
            -0.0057733115
          ],
          [
            "ick",
            -0.00034154154
          ]
        ]
      },
      "After this character relates a story about how he didn't know the proper way to use a wheelbarrow, he tells of how a captain dining with his father mistakenly rubbed his hands in a punch bowl. This \"sea Prince of Wales\" leaves his home by hiding out in a canoe near a coral reef, and he is mistakenly called \"Hedgehog\" by a character who offers him a ninetieth lay, a partner of Bildad named Peleg. A": {
        "guess": "Queequeg",
        "confidence": [
          [
            "Que",
            -0.34479463
          ],
          [
            "e",
            -8.299462e-06
          ],
          [
            "que",
            -2.8160932e-06
          ],
          [
            "g",
            -3.3213026e-05
          ]
          ]
        }
          	
You could imagine other ways of using the word piece probabilities
rather than just taking the arithmetic mean of the log probs (which is
what the code is currently doing).  As before, the goal is to be
creative and to understand the data.  Good luck!

Because we have already tested your guesser, we will not be retraining
your Guesser nor your Buzzer.  So make sure that the ``save`` and ``load`` functions
work correctly (for both the Guesser and the Buzzer).  Also make sure
that you've trained it on as much data as possible.  You'll need to
modify `params.py` so that whatever works best for you is the
default.  

These submissions will be the foundation for our in-class exposition
game, so please do try to do this extra credit, as this will be one of
the most fun opportunities we'll have for extra credit in the class.

Hints
-

1.  Don't use all of the data, especially at first.  Use the _limit_
    command line argument (as in the above example).  Indeed, you
    might be able to improve accuracy by *excluding* some of the data.
1.  On a related note, don't create a [dense matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.todense.html) out of tf-idf counts.  By default, the tf-idf libraries will create a sparse matrix where only the non-zero elements are allocated memory.  A dense matrix will require way too much space (e.g., if you have 25k terms and 300k documents, that's going to be around 50 GiB, and that's going to be too big for most laptops and certainly for Gradescope).  Any of the operations that you need to do you can do with the sparse matrix.
3.  In case you see an error that your submission timed out on Gradescope, that means that your code needs to be simplified. 
    This is essential for your  code to work on Gradescope, so think of ways
    you can optimize your code.  Another issue if
    if you're trying to create the matrix one row at a time; it's possible to
    do it in batch, and that will speed things up.
2.  If the guesser submission says that your pickle file is missing, this means that the guesser training errored out and didn't generate the file.      
2.  Another problem with the (extra credit) submission might be that your pickle file (how your vectorizer / matrix is saved) is too large (Gradescope has a 100MB limit).  Remember that your tf-idf representation is a matrix.  It could be that your tf-idf representation
    is too wide (too many terms) or too tall (too many documents).  You had to
    deal with this before in your previous tf-idf homework.  (Think
    about building your vocabulary!  There are similar options in ``sklearn``)
2.  tf-idf representations do not know anything about syntax or part of
    speech.  You could add features to correct some of these problems.  (This
    is just for the extra credit!)    
6.  Don't forget about the definition of what a token is and how it's flexible.  The sklearn
    tokenizer does support n-grams, which may help you in the extra credit (but consume more
    memory):
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html 
5.  The buzzer leaderboard will report both accuracy and buzz ratio.  Both are important, as you can only decide if a guess is
    correct if the correct guess is an option: you can get 100%
    accuracy on the buzzer if all of the guesses are wrong... but your
    buzz position will be horrible.
7.  *Do not focus on buzzer accuracy to early*!  When your guesser is broken, all of
    the guesses will be wrong and you'll trivially get perfect buzz accuracy
    (always wait).  Unless you're going for going after extra credit, you should pay attention to precision and recall (which are specific to the guesser).
8.  That said, accuracy comes from the buzzer; if you have a bad
    accuracy score despite updating the guesser, it's possible that
    the pickle for your buzzer has not been updated and is looking for
    the wrong features (or is miscalibrated).  Focusing on buzz
    position is more worthwhile.
9.  If you find that things are taking too long (things are timing out on Gradescope), implement the
    ``batch_guess`` function to guess on many examples at once.
10.  For the extra credit, we strongly recommend you use the GPT guesser **in conjunction** with the tf-idf guesser.  Make the GPT guesser the primary guesser, and then you can add additional tf-idf features to that.  It should improve from what you were able to do with the GPT guesser alone.  You're also welcome to add additional guessers / information (like from Wikipedia).
9.  Once you've completed the required part of the homework and you're
    trying to increase the recall further, you can investigate
    changing the dimensions of the vectorization: what normalization
    is applied to the words, what data are included, or looking at
    n-grams.  Also don't forget
    about the wiki pages:
    https://drive.google.com/file/d/1-AhjvqsoZ01gz7EMt5VmlCnVpsE96A5n/view?usp=share_link. The file is under the `data` folder on gradescope.
11.  If you get an error ``max_df corresponds to < documents than min_df``, think about what this means.  It's complaining that you're excluding all tokens by setting thresholds that would exclude **everything**.  There are two likely causes for this:
     - One cause is tricky.  For the unit tests, we tell you to have ``max_df=1.0, min_df=0.0`` (i.e., let everything in).  But if you instead type ``max_df=1, min_df=0``, then it will exclude everything appearing in more than one document.  This is because interprets not specifying them as float (which are interpreted as frequency) but rather as ints (which are interpreted as number of documents).  **Important:*** The constructor defaults (``min_df=10`` and ``max_df=0.4``) are different from behavior to be tested.  You can and should set your tokenizer thresholds based on performance **differently** from how you set them to pass the unit tests.
     -  The defaults might also trigger if your limit flag is too small.  In other words, if you're using 25 or fewer documents, then 10 will be the same as 0.4.  This is of course the case on the unit tests, where you should be using ``max_df=1.0, min_df=0.0``.
    

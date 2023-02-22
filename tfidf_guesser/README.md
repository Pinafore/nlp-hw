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

You'll turn in your code on Gradescope.

What you have to do
----

Coding (15 points in the tfidf_guesser.py):

1.  (Optional) Store necessary data in the constructor so you can do classification later.
1.  Modify the _train_ function so that the class stores what it needs to store to guess at what the answer is.
1.  Modify the _call_ function so that it finds the closest indicies (in terms of *cosine* similarity) to the query.

Analysis (5 points):

1.  What is the role of the number of training points to accuracy?
1.  What answers get confused with each other most easily?
1.  Compute precision and recall as you increase the number of guesses.

Accuracy (10 points): How well you do on the leaderboard.

What you don't have to do
-------

You don't have to (and shouldn't!) compute tf-idf yourself.  We did that in
a previous homework, so you can leave that to the professionals.  We encourage
you to use the tf-idf vectorizer: play around with different settings of the
paramters.  You probably shouldn't modify it, but it's probably useful to
understand it for future homeworks (you'll need to write/call code like it in
the future).


What to turn in
-

1.  Submit your _tfidf_guesser.py_ file
2.  If you create new features (or reuse features from the feature engineering
homework), also upload your _params.py_ and _features.py_ files.
3.  Submit your _analysis.pdf_ file (no more than one page; pictures
    are better than text)

Extra Credit
=
You can get extra credit for by submitting your system on Dynabench (assuming
we can get it up in time ... watch Piazza for announcements).


Example
-

This is an example of what your code (tfidf_guesser.py) output should look like:
```
> python3 eval.py --evaluate=guesser --question_source=gzjson
--questions=data/qanta.guessdev.json.gz --limit=500

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

Hints
-

1.  Don't use all of the data, especially at first.  Use the _limit_
    command line argument (as in the above example).
2.  You probably want to tune tf-idf parameters.  Play around with what works well!
3.  You can add additional data if you want.  You can create an additional
    guesser if you want to keep it separate from the tfidf_guesser.  If you do
    that, make sure to upload that file too.
4.  tf-idf representations do not know anything about syntax or part of
    speech.  You could add features to correct some of these problems.
5.  The leaderboard will report both retrieval accuracy and final buzz
    accuracy.  Both are important, as you can only decide if a guess is
    correct if the correct guess is an option.
6.  Don't forget about the definition of what a token is and how it's flexible.  The sklearn
    tokenizer does support n-grams, which may help you (but consume more
    memory):
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html 

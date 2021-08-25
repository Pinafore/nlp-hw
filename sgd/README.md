Stochastic Gradient Descent
=

Overview
--------

In this homework you'll implement a stochastic gradient ascent for
logistic regression and you'll apply it to the task of determining
whether an answer to a question is correct or not.

This will be slightly more difficult than the last homework (the difficulty
will slowly ramp upward).  You should not use any libraries that implement any
of the functionality of logistic regression for this assignment; logistic
regression is implemented in scikit learn, pytorch, and many other places, but
you should do everything by hand now.  You'll be able to use library
implementations of logistic regression.

You'll turn in your code on Gradescope.  This assignment is worth 30 points.

What you have to do
----

Coding (25 points):

1. Understand how the code is creating feature vectors (this will help you
code the solution and to do the later analysis).  You don't actually need to
write any code for this, however.

2. (Optional) Store necessary data in the constructor so you can do
classification later.

3. You'll likely need to write some code to get the best/worst features (see
below).

3. Modify the _sg update_ function to perform non-regularized updates.

Analysis (5 points):

1. What is the role of the learning rate?
2. How many datapoints (or multiple passes over the data) do you need to
complete for the *model* to stabilize?  The various metrics can give you clues
about what the model is doing, but no one metric is perfect.
3. What features are the best predictors of each class?  How (mathematically)
did you find them?
4. What features are the poorest predictors of classes?  How (mathematically)
did you find them?

Extra credit:

1. Use a schedule to update the learning rate.
    - Supply an appropriate argument to step parameter
    - Support it in your _sg update_
    - Show the effect in your analysis document
2.  Modify the _sg update_ function to perform [lazy regularized updates](https://lingpipe.files.wordpress.com/2008/04/lazysgdregression.pdf), which only update the weights of features when they appear in an example.
    - Show the effect in your analysis document 
    
Caution: When implementing extra credit, make sure your implementation of the
regular algorithms doesn't change.

What to turn in
-

1. Submit your _sgd.py_ file (include your name at the top of the source)
1. Submit your _analysis.pdf_ file
    - no more than one page (NB: This is also for the extra credit.  To minimize effort for the grader, you'll need to put everything on a page.  Take this into account when selecting if/which extra credit to do...think of the page requirement like a regularizer).
    - pictures are better than text
    - include your name at the top of the PDF

Unit Tests
=

I've provided unit tests based on the example that we worked through
in class.  Before running your code on read data, make sure it passes
all of the unit tests.

```
cs244-33-dhcp:logreg jbg$ python tests.py
.[ 0.  0.  0.  0.  0.]
[ 1.  4.  3.  1.  0.]
F
======================================================================
FAIL: test_unreg (__main__.TestKnn)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests.py", line 22, in test_unreg
    self.assertAlmostEqual(b1[0], .5)
AssertionError: 0.0 != 0.5 within 7 places

----------------------------------------------------------------------
Ran 2 tests in 0.001s

FAILED (failures=1)
```

Example
-

This is an example of what your runs should look like:
```
python3 sgd.py 
Loaded 813 items from vocab data/small_guess.vocab
Read in 16438 train and 962 test
Update 1	TProb -11272.713595	HProb -644.791326	TAcc 0.582553	HAcc 0.787942
Update 101	TProb -10951.708385	HProb -547.249129	TAcc 0.595206	HAcc 0.787942
Update 201	TProb -10889.160007	HProb -539.922408	TAcc 0.604575	HAcc 0.782744
Update 301	TProb -10854.428640	HProb -577.292161	TAcc 0.605487	HAcc 0.744283
Update 401	TProb -10629.755560	HProb -520.975509	TAcc 0.623008	HAcc 0.787942
Update 501	TProb -10538.804412	HProb -514.693129	TAcc 0.635114	HAcc 0.788981
Update 601	TProb -10521.839479	HProb -500.047394	TAcc 0.637182	HAcc 0.787942
Update 701	TProb -10453.196143	HProb -503.612648	TAcc 0.641684	HAcc 0.782744
Update 801	TProb -10437.154837	HProb -518.424938	TAcc 0.639616	HAcc 0.774428
Update 901	TProb -10402.081967	HProb -506.207993	TAcc 0.651539	HAcc 0.787942
Update 1001	TProb -10348.939892	HProb -502.169692	TAcc 0.648315	HAcc 0.787942
Update 1101	TProb -10366.327293	HProb -498.274113	TAcc 0.643570	HAcc 0.784823
Update 1201	TProb -10428.542079	HProb -538.884260	TAcc 0.636635	HAcc 0.746362
Update 1301	TProb -10363.182698	HProb -494.847738	TAcc 0.648072	HAcc 0.787942
Update 1401	TProb -10234.012740	HProb -493.917454	TAcc 0.654216	HAcc 0.779626
Update 1501	TProb -10346.204936	HProb -535.606174	TAcc 0.640467	HAcc 0.753638
Update 1601	TProb -10482.549442	HProb -486.969978	TAcc 0.644421	HAcc 0.787942
Update 1701	TProb -10597.077223	HProb -486.347296	TAcc 0.637608	HAcc 0.790021
Update 1801	TProb -10162.238391	HProb -510.956940	TAcc 0.652573	HAcc 0.774428
Update 1901	TProb -10019.696102	HProb -485.680817	TAcc 0.668694	HAcc 0.788981
Update 2001	TProb -10075.765879	HProb -481.928838	TAcc 0.663645	HAcc 0.795218
Update 2101	TProb -10114.072404	HProb -506.497198	TAcc 0.654946	HAcc 0.762994
Update 2201	TProb -10066.648578	HProb -491.262360	TAcc 0.656284	HAcc 0.785863
Update 2301	TProb -9986.156662	HProb -481.544781	TAcc 0.669546	HAcc 0.790021
Update 2401	TProb -9961.308369	HProb -485.837950	TAcc 0.669059	HAcc 0.787942
Update 2501	TProb -10122.608577	HProb -482.687852	TAcc 0.665470	HAcc 0.792100
Update 2601	TProb -9915.254905	HProb -480.191499	TAcc 0.671310	HAcc 0.795218
Update 2701	TProb -10236.593808	HProb -512.014789	TAcc 0.641319	HAcc 0.770270
Update 2801	TProb -9899.679092	HProb -482.503681	TAcc 0.673257	HAcc 0.792100
Update 2901	TProb -9939.258589	HProb -482.362967	TAcc 0.671371	HAcc 0.787942
Update 3001	TProb -10299.168348	HProb -494.079103	TAcc 0.635479	HAcc 0.777547
Update 3101	TProb -9799.735736	HProb -480.820906	TAcc 0.679584	HAcc 0.790021
Update 3201	TProb -9844.953347	HProb -481.645334	TAcc 0.671432	HAcc 0.788981
Update 3301	TProb -9860.861187	HProb -481.988264	TAcc 0.668938	HAcc 0.785863
Update 3401	TProb -10021.195859	HProb -496.384653	TAcc 0.660299	HAcc 0.775468
Update 3501	TProb -9781.283161	HProb -479.328495	TAcc 0.679462	HAcc 0.787942
Update 3601	TProb -9815.387024	HProb -480.954085	TAcc 0.679462	HAcc 0.792100
Update 3701	TProb -10161.440258	HProb -496.613754	TAcc 0.645821	HAcc 0.776507
Update 3801	TProb -10175.293309	HProb -497.697455	TAcc 0.647646	HAcc 0.777547
Update 3901	TProb -9961.136377	HProb -484.397557	TAcc 0.666444	HAcc 0.785863
Update 4001	TProb -9769.231596	HProb -479.460037	TAcc 0.678246	HAcc 0.791060
Update 4101	TProb -9740.451742	HProb -474.615068	TAcc 0.683295	HAcc 0.794179
Update 4201	TProb -9982.003198	HProb -488.400693	TAcc 0.662003	HAcc 0.787942
Update 4301	TProb -9694.257558	HProb -474.670089	TAcc 0.685302	HAcc 0.793139
Update 4401	TProb -10072.035266	HProb -489.605057	TAcc 0.655980	HAcc 0.780665
Update 4501	TProb -9971.923933	HProb -481.919264	TAcc 0.666991	HAcc 0.788981
Update 4601	TProb -9675.541962	HProb -476.207919	TAcc 0.682747	HAcc 0.792100
Update 4701	TProb -9704.672248	HProb -472.221312	TAcc 0.686154	HAcc 0.791060
Update 4801	TProb -9674.826224	HProb -479.044614	TAcc 0.680253	HAcc 0.791060
Update 4901	TProb -9741.696172	HProb -472.178574	TAcc 0.679219	HAcc 0.787942
Update 5001	TProb -9610.540732	HProb -476.183097	TAcc 0.685972	HAcc 0.792100
Update 5101	TProb -9806.416855	HProb -487.537888	TAcc 0.671858	HAcc 0.787942
Update 5201	TProb -9649.319502	HProb -469.220368	TAcc 0.686945	HAcc 0.793139
Update 5301	TProb -9604.259087	HProb -468.771346	TAcc 0.686093	HAcc 0.792100
Update 5401	TProb -9682.240491	HProb -469.916405	TAcc 0.681166	HAcc 0.780665
Update 5501	TProb -10393.808385	HProb -479.172926	TAcc 0.616255	HAcc 0.791060
Update 5601	TProb -9504.002565	HProb -469.507121	TAcc 0.694367	HAcc 0.793139
Update 5701	TProb -9769.935370	HProb -489.839836	TAcc 0.675447	HAcc 0.788981
Update 5801	TProb -9513.363745	HProb -463.364214	TAcc 0.699112	HAcc 0.794179
Update 5901	TProb -9852.728596	HProb -467.043003	TAcc 0.669911	HAcc 0.793139
Update 6001	TProb -9810.014102	HProb -500.010689	TAcc 0.671189	HAcc 0.787942
Update 6101	TProb -9505.222699	HProb -478.038825	TAcc 0.689622	HAcc 0.790021
Update 6201	TProb -9557.074781	HProb -468.159783	TAcc 0.693880	HAcc 0.793139
Update 6301	TProb -9447.239605	HProb -468.106529	TAcc 0.702701	HAcc 0.793139
Update 6401	TProb -9549.542312	HProb -492.075581	TAcc 0.692603	HAcc 0.787942
Update 6501	TProb -9533.320324	HProb -482.033603	TAcc 0.693333	HAcc 0.792100
Update 6601	TProb -9949.355135	HProb -467.774372	TAcc 0.660117	HAcc 0.796258
Update 6701	TProb -9440.618893	HProb -470.979749	TAcc 0.697469	HAcc 0.792100
Update 6801	TProb -9430.964811	HProb -470.369234	TAcc 0.702701	HAcc 0.791060
Update 6901	TProb -9368.895176	HProb -466.252298	TAcc 0.707629	HAcc 0.793139
Update 7001	TProb -9466.880267	HProb -463.715218	TAcc 0.704891	HAcc 0.795218
Update 7101	TProb -9393.822467	HProb -483.707565	TAcc 0.694123	HAcc 0.790021
Update 7201	TProb -9399.641249	HProb -476.316536	TAcc 0.698929	HAcc 0.790021
Update 7301	TProb -9414.788486	HProb -472.502142	TAcc 0.695523	HAcc 0.794179
Update 7401	TProb -9457.789781	HProb -464.579313	TAcc 0.691082	HAcc 0.795218
Update 7501	TProb -9617.851255	HProb -472.127170	TAcc 0.679949	HAcc 0.795218
Update 7601	TProb -9394.712127	HProb -462.788468	TAcc 0.699173	HAcc 0.799376
Update 7701	TProb -9322.883190	HProb -476.445209	TAcc 0.703188	HAcc 0.792100
Update 7801	TProb -9368.808346	HProb -460.447200	TAcc 0.709089	HAcc 0.793139
Update 7901	TProb -9425.068410	HProb -457.171225	TAcc 0.702701	HAcc 0.798337
Update 8001	TProb -9313.126363	HProb -458.723437	TAcc 0.712800	HAcc 0.795218
Update 8101	TProb -9959.306025	HProb -460.717542	TAcc 0.658231	HAcc 0.795218
Update 8201	TProb -9361.775804	HProb -479.124379	TAcc 0.697348	HAcc 0.790021
Update 8301	TProb -9243.607424	HProb -470.146194	TAcc 0.711705	HAcc 0.792100
Update 8401	TProb -9241.348054	HProb -474.856775	TAcc 0.705621	HAcc 0.793139
Update 8501	TProb -9519.673946	HProb -458.428376	TAcc 0.695766	HAcc 0.794179
Update 8601	TProb -9197.905596	HProb -477.435286	TAcc 0.711461	HAcc 0.793139
Update 8701	TProb -9280.658834	HProb -493.098973	TAcc 0.703979	HAcc 0.787942
Update 8801	TProb -9301.316781	HProb -488.335582	TAcc 0.704648	HAcc 0.792100
Update 8901	TProb -9336.923499	HProb -479.773378	TAcc 0.703796	HAcc 0.793139
Update 9001	TProb -9264.790947	HProb -491.275988	TAcc 0.703614	HAcc 0.791060
Update 9101	TProb -9332.465693	HProb -472.029784	TAcc 0.703005	HAcc 0.796258
Update 9201	TProb -9264.276287	HProb -468.601028	TAcc 0.710245	HAcc 0.797297
Update 9301	TProb -9197.650525	HProb -482.562227	TAcc 0.708663	HAcc 0.792100
Update 9401	TProb -9186.151657	HProb -463.604077	TAcc 0.713104	HAcc 0.796258
Update 9501	TProb -9555.112345	HProb -466.692083	TAcc 0.688587	HAcc 0.787942
Update 9601	TProb -9491.875725	HProb -495.817954	TAcc 0.691142	HAcc 0.794179
Update 9701	TProb -9267.332970	HProb -469.987582	TAcc 0.706412	HAcc 0.791060
Update 9801	TProb -9250.571849	HProb -475.446400	TAcc 0.707750	HAcc 0.794179
Update 9901	TProb -10235.318236	HProb -560.481287	TAcc 0.665166	HAcc 0.787942
Update 10001	TProb -9251.638819	HProb -492.617673	TAcc 0.702944	HAcc 0.792100
Update 10101	TProb -9268.031354	HProb -477.799736	TAcc 0.704039	HAcc 0.794179
Update 10201	TProb -9510.842413	HProb -464.611279	TAcc 0.689682	HAcc 0.796258
Update 10301	TProb -9204.988685	HProb -472.745397	TAcc 0.712070	HAcc 0.793139
Update 10401	TProb -9532.105503	HProb -461.766053	TAcc 0.689013	HAcc 0.799376
Update 10501	TProb -9215.461622	HProb -462.708825	TAcc 0.712678	HAcc 0.793139
Update 10601	TProb -9123.328056	HProb -465.349481	TAcc 0.713590	HAcc 0.794179
Update 10701	TProb -9130.910814	HProb -472.987339	TAcc 0.712009	HAcc 0.793139
Update 10801	TProb -9191.376503	HProb -478.422319	TAcc 0.705499	HAcc 0.792100
Update 10901	TProb -9268.510916	HProb -458.543893	TAcc 0.707020	HAcc 0.795218
Update 11001	TProb -9207.658492	HProb -474.808341	TAcc 0.701545	HAcc 0.793139
Update 11101	TProb -9314.905434	HProb -474.145486	TAcc 0.697956	HAcc 0.795218
Update 11201	TProb -9152.960043	HProb -452.748038	TAcc 0.712130	HAcc 0.796258
Update 11301	TProb -9071.136422	HProb -462.496079	TAcc 0.716815	HAcc 0.796258
Update 11401	TProb -9132.694464	HProb -454.426296	TAcc 0.717910	HAcc 0.802495
Update 11501	TProb -9432.567067	HProb -495.218628	TAcc 0.695888	HAcc 0.792100
Update 11601	TProb -9068.728832	HProb -467.148975	TAcc 0.714260	HAcc 0.795218
Update 11701	TProb -9042.302479	HProb -464.217591	TAcc 0.714199	HAcc 0.792100
Update 11801	TProb -9026.261790	HProb -461.209922	TAcc 0.718579	HAcc 0.798337
Update 11901	TProb -9138.763149	HProb -476.093186	TAcc 0.707203	HAcc 0.795218
Update 12001	TProb -9086.013359	HProb -462.358159	TAcc 0.713286	HAcc 0.797297
Update 12101	TProb -9125.802991	HProb -462.152996	TAcc 0.710488	HAcc 0.798337
Update 12201	TProb -9096.677743	HProb -459.551441	TAcc 0.711461	HAcc 0.799376
Update 12301	TProb -9176.525791	HProb -458.657829	TAcc 0.707081	HAcc 0.797297
Update 12401	TProb -9172.368983	HProb -459.549285	TAcc 0.708420	HAcc 0.798337
Update 12501	TProb -9034.436164	HProb -464.453495	TAcc 0.718883	HAcc 0.797297
Update 12601	TProb -9228.187252	HProb -492.440624	TAcc 0.704344	HAcc 0.790021
Update 12701	TProb -9052.452201	HProb -458.660618	TAcc 0.717788	HAcc 0.799376
Update 12801	TProb -9269.126843	HProb -494.882322	TAcc 0.703309	HAcc 0.792100
Update 12901	TProb -9123.309514	HProb -475.165890	TAcc 0.714381	HAcc 0.795218
Update 13001	TProb -9275.219934	HProb -477.482267	TAcc 0.708420	HAcc 0.796258
Update 13101	TProb -9029.545050	HProb -466.764862	TAcc 0.713530	HAcc 0.796258
Update 13201	TProb -9239.819204	HProb -447.795645	TAcc 0.709819	HAcc 0.805613
Update 13301	TProb -8996.640829	HProb -469.959898	TAcc 0.715476	HAcc 0.792100
Update 13401	TProb -9562.580162	HProb -450.955564	TAcc 0.685120	HAcc 0.793139
Update 13501	TProb -8943.605792	HProb -457.418290	TAcc 0.724297	HAcc 0.799376
Update 13601	TProb -9061.008171	HProb -465.692010	TAcc 0.712921	HAcc 0.796258
Update 13701	TProb -9562.835145	HProb -503.603317	TAcc 0.685363	HAcc 0.791060
Update 13801	TProb -8926.047473	HProb -454.433834	TAcc 0.729286	HAcc 0.797297
Update 13901	TProb -9536.283470	HProb -447.809983	TAcc 0.685667	HAcc 0.804574
Update 14001	TProb -8928.868504	HProb -451.337124	TAcc 0.726731	HAcc 0.797297
Update 14101	TProb -9019.224821	HProb -445.824380	TAcc 0.727339	HAcc 0.803534
Update 14201	TProb -9157.828477	HProb -484.143133	TAcc 0.705439	HAcc 0.794179
Update 14301	TProb -9000.403981	HProb -473.446078	TAcc 0.716754	HAcc 0.793139
Update 14401	TProb -9099.981813	HProb -483.380575	TAcc 0.708359	HAcc 0.791060
Update 14501	TProb -8930.920712	HProb -456.448959	TAcc 0.730868	HAcc 0.799376
Update 14601	TProb -9006.953953	HProb -454.232746	TAcc 0.726427	HAcc 0.798337
Update 14701	TProb -9234.764026	HProb -493.797654	TAcc 0.699659	HAcc 0.792100
Update 14801	TProb -8915.536498	HProb -462.534353	TAcc 0.725514	HAcc 0.799376
Update 14901	TProb -9272.660593	HProb -495.815509	TAcc 0.698686	HAcc 0.793139
Update 15001	TProb -9030.695135	HProb -476.227550	TAcc 0.716693	HAcc 0.799376
Update 15101	TProb -8990.011703	HProb -463.203985	TAcc 0.716693	HAcc 0.801455
Update 15201	TProb -9080.732553	HProb -460.644469	TAcc 0.714016	HAcc 0.800416
Update 15301	TProb -8987.963668	HProb -466.365733	TAcc 0.720708	HAcc 0.798337
Update 15401	TProb -8924.602178	HProb -459.479498	TAcc 0.723993	HAcc 0.799376
Update 15501	TProb -8961.858540	HProb -450.342188	TAcc 0.724297	HAcc 0.802495
Update 15601	TProb -9190.046969	HProb -491.825560	TAcc 0.705013	HAcc 0.795218
Update 15701	TProb -9078.914513	HProb -448.071562	TAcc 0.713651	HAcc 0.803534
Update 15801	TProb -8966.862965	HProb -472.840226	TAcc 0.713955	HAcc 0.795218
Update 15901	TProb -8797.661484	HProb -453.533653	TAcc 0.727582	HAcc 0.800416
Update 16001	TProb -8897.925665	HProb -457.135901	TAcc 0.716511	HAcc 0.798337
Update 16101	TProb -8823.984145	HProb -447.882049	TAcc 0.736160	HAcc 0.796258
Update 16201	TProb -8931.043162	HProb -463.082295	TAcc 0.718031	HAcc 0.793139
Update 16301	TProb -8926.584410	HProb -454.446049	TAcc 0.726548	HAcc 0.796258
Update 16401	TProb -8983.403735	HProb -456.031943	TAcc 0.723324	HAcc 0.797297
```

Hints
-

1.  As with the previous assignment, make sure that you debug on small
    datasets first (I've provided _toy text_ in the data directory to get you started).
1.  Certainly make sure that you do the unregularized version first
    and get it to work well.
1.  Use numpy functions whenever you can to make the computation faster.



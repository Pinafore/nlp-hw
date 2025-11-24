
from buzzer import BuzzerParameters, Buzzer

from sklearn.linear_model import LogisticRegression

class ThresholdParameters(BuzzerParameters):
    def __init__(self, customized_params=None):
        BuzzerParameters.__init__(self)
        self.name = "threshold_buzzer"
        if customized_params:
            self.params += customized_params
        else:
            threshold_params = [("threshold", float, 0.5, "If guesser has confidence over this value, we buzz")]
            self.params += threshold_params

class ThresholdBuzzer(Buzzer):
    def __init__(self, threshold, run_length):
        super().__init__(filename="", run_length=run_length, num_guesses=1)

        self.threshold = threshold

        # This is needed for error script not to error, could probably be more elegantly fixed
        self._classifier = LogisticRegression()
        self._classifier.coef_ = [[]]        

    def load(self):
        None
        
    def predict(self, questions):
        assert len(self._features) == len(self._questions), "Features not built.  Did you run build_features?"

        predictions = []
        threshold_feature = None
        for feat_vec in self._features:
            if threshold_feature is None:
                assert sum(1 for x in feat_vec if x.endswith("confidence")) == 1, "Too many confidences"
                for feat_name in feat_vec:
                    if feat_name.endswith("confidence"):
                        threshold_feature = feat_name
            assert threshold_feature is not None
                        
            if feat_vec[threshold_feature] > self.threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions, None, self._features, self._correct, self._metadata

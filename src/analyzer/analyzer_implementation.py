import pandas as pd

from .analyzer_abstraction import AnalyzerAbstraction


class AnalyzerImplementation(AnalyzerAbstraction):
    def __init__(self, model, cols, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.cols = cols

    def analyze(self, stats_arr):
        """
        Analyze stats
        Args:
            stats_arr: array with calculated stats

        Returns:
            explosion probability
        """
        X = pd.DataFrame([stats_arr])[self.cols]
        X = X.astype({x: "float" for x in X.columns})

        return self.model.predict_proba(X[self.cols])[:, 1]

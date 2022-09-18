import logging

import pandas as pd

from src.ingest_gt import load_gt, create_gt
from src.io_manager import load_pickle, load_list
from src.model_trainer.model_trainer_implementation import ModelTrainerImplementation
from .model_evaluator_abstraction import ModelEvalAbstraction


class ModelEvalImplementation(ModelEvalAbstraction):

    def __init__(self, labels_path, model_path, cols_path, eval_result_path):
        self.labels_path = labels_path
        self.model_path = model_path
        self.cols_path = cols_path
        self.eval_result_path = eval_result_path

        self.stats_arr = []

    def stats_callback(self, stats):
        """
        Callback method to use with StreamInputter
        Args:
            stats (dict): dictionary with `{'feature_name': value}`

        """
        self.stats_arr.append(stats)

    def eval_model(self):
        """
        Train model with stream read wholly. Output trained model, features and predictions
        """
        X = pd.DataFrame(self.stats_arr).set_index("approximate_timing")
        X = X.astype({x: "float" for x in X.columns})
        y = load_gt(self.labels_path)
        y = create_gt(y, X)

        cols = load_list(self.cols_path)
        X = X[cols]

        model = load_pickle(self.model_path)

        results = ModelTrainerImplementation.score_callback(model, X, y.astype(int))
        print("Model evaluation results:")
        print(results)

        self.write_resulting_df(X)

    def write_resulting_df(self, df):
        """Write predictions made by evaluated model into csv for eventual further inspection

        Args:
            df (pd.DataFrame): data with features and predictions

        Returns:

        """
        df.to_csv(self.eval_result_path)
        print(f"New predictions on whole data written to {self.eval_result_path}")


import logging
import os
import tempfile
import optuna

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, \
    precision_recall_curve, auc
from xgboost import XGBClassifier, plot_importance

from src.ingest_gt import load_gt, create_gt
from src.io_manager import write_list, load_pickle
from src.io_manager import write_pickle
from .model_trainer_abstraction import ModelTrainerAbstraction


class ModelTrainerImplementation(ModelTrainerAbstraction):

    def __init__(self, labels_path, model_path, cols_path, train_result_path, model_id):
        self.labels_path = labels_path
        self.model_path = model_path
        self.cols_path = cols_path
        self.train_result_path = train_result_path
        self.model_id = model_id

        self.stats_arr = []

    def stats_callback(self, stats):
        """
        Callback method to use with StreamInputter
        Args:
            stats (dict): dictionary with `{'feature_name': value}`

        """
        self.stats_arr.append(stats)

    def train_model(self):
        """
        Train model with stream read wholly. Output trained model, features and predictions
        """
        X = pd.DataFrame(self.stats_arr).set_index("approximate_timing")
        X = X.astype({x: "float" for x in X.columns})
        X['rand'] = np.random.random(X.shape[0])
        y = load_gt(self.labels_path)
        y = create_gt(y, X)

        X_train, y_train, X_test, y_test = self.split_data(X, y)
        model = self.fit_model(X_train, y_train)

        results = self.score_callback(model, X_test, y_test.astype(int))
        logging.info(results)

        model, cols = self.eval_shaps(model, X_train, y_train, X_test, y_test)
        write_pickle(model, self.model_path)
        write_list(cols, self.cols_path)

        # HP tuning
        experiment_name = f"model{self.model_id}"
        best_params, model, cols = self.tune_hps(X_train, y_train, X_test, y_test, experiment_name)

        print("Final results on test set:", ModelTrainerImplementation.score_callback(model, X_test[cols], y_test))

        write_pickle(model, self.model_path)
        write_list(cols, self.cols_path)

        X['prediction'] = model.predict_proba(X[cols])[:, 1]
        X['label'] = y.astype(int).values

        self.write_resulting_df(X)

        logging.info('Model has been successfully trained')

    @staticmethod
    def split_data(X, y, max_test_size=0.23):
        """Split data into train and test sets

        Args:
            X (pd.DataFrame): features
            y (pd.Series): labels
            max_test_size (float): maximum size of test set

        Returns:
            pd.DataFrame, pd.Series, pd.DataFrame, pd.Series: train features, train labels, test features, test labels

        """
        n = X.shape[0]
        gap = 300  # points

        n_train = int((1 - max_test_size) * n)
        n_test = n - n_train - gap

        X_train = X[:n_train]
        y_train = y[:n_train]

        X_test = X[-n_test:]
        y_test = y[-n_test:]

        return X_train, y_train, X_test, y_test

    @staticmethod
    def fit_model(X, y):
        """Train the XGBoost model

        Args:
            X (pd.DataFrame): features
            y (pd.Series): labels

        Returns:
            xgboost.XGBClassifier: trained model

        """
        model = XGBClassifier()

        model.fit(X, y.astype(int).values)
        return model

    @staticmethod
    def score_callback(clf, X_t, y_t, thr=.5):
        """Score model's performance

        Args:
            clf (xgboost.XGBClassifier): trained model
            X_t (pd.DataFrame): test features
            y_t (pd.Series): test labels
            thr (float): probability threshold, such that predictions `>=thr` are trated as `1`, `0` otherwise

        Returns:
            dict: dict with the evaluation metrics on the model with given data

        """
        y_p = clf.predict_proba(X_t)[:, 1]

        y_p_raw = y_p.copy()

        y_p[y_p < thr] = 0
        y_p[y_p >= thr] = 1

        # Data to plot precision - recall curve
        precision, recall, thresholds = precision_recall_curve(y_t, y_p)

        (p, r, f, _) = precision_recall_fscore_support(y_t, y_p)

        # auc = roc_auc_score(y_t, y_p_raw)

        # Use AUC function to calculate the area under the curve of precision recall curve
        auc_precision_recall = auc(recall, precision)
        acc = accuracy_score(y_t, y_p)

        tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()

        return {
            'acc': acc,
            'auc': auc_precision_recall,
            'p_0': p[0],
            'r_0': r[0],
            'f_0': f[0],
            'p_1': p[1],
            'r_1': r[1],
            'f_1': f[1],
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }

    @staticmethod
    def eval_shaps(model, X_train, y_train, X_test, y_test, plot_stuff=True):
        """Evaluate feature contribution (SHAP, but not really) and select the best feature sets

        Args:
            model (xgboost.XGBClassifier): fitted predictor model
            X_train (pd.DataFrame): Training feature set
            y_train (pd.Series): Training label set
            X_test (pd.DataFrame): Test feature set
            y_test (pd.Series): Test label set

        Returns:
            xgboost.XGBClassifier, list: newly fitted estimator and best feature set

        """
        if plot_stuff:
            plot_importance(model)

        thresholds = np.sort(model.feature_importances_)

        feat_df = pd.DataFrame({"Importance": model.feature_importances_, "name": X_train.columns}).set_index("name")
        print(feat_df.head())

        max_acc = 0
        thr = 0

        was_zero = False

        for thresh in thresholds:
            if (thresh == 0) and was_zero:
                continue
            elif thresh == 0:
                was_zero = True
            # select features using threshold
            selection = SelectFromModel(model, threshold=thresh, prefit=True)
            select_X_train = selection.transform(X_train)
            # train model
            selection_model = XGBClassifier()
            selection_model.fit(select_X_train, y_train)
            # eval model
            select_X_test = selection.transform(X_test)
            y_pred = selection_model.predict(select_X_test)
            predictions = [round(value) for value in y_pred]
            (p, r, f, _) = precision_recall_fscore_support(y_test, predictions)
            # accuracy = accuracy_score(y_test, predictions)
            if f[1] > max_acc:
                max_acc = f[1]
                thr = thresh
            print("Thresh=%.3f, n=%d, F-score: %.2f%%" % (thresh, select_X_train.shape[1], f[1] * 100.0))

        # train model
        selection_model = XGBClassifier()
        cols = feat_df[feat_df.Importance >= thr].index
        cols = [x for x in cols if x != "rand"]

        selection_model.fit(X_train[cols], y_train)
        # eval model
        y_pred = selection_model.predict(X_test[cols])
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thr, X_train[cols].shape[1], accuracy * 100.0))
        if plot_stuff:
            plot_importance(selection_model)

        print("Champion cols:", cols)

        return selection_model, cols

    def write_resulting_df(self, df):
        """Write predictions made by newly trained model into csv for eventual further inspection

        Args:
            df (pd.DataFrame): data with features and predictions

        Returns:

        """
        df.to_csv(self.train_result_path)
        print(f"New predictions on whole data written to {self.train_result_path}")

    @staticmethod
    def objective(trial, X_train, y_train, X_test, y_test, tmpdirname):
        """ Callback for hyperparameter tuning for XGBoost

        Args:
            trial (optuna.trial.Trial): Optuna trial object
            X_train (pd.DataFrame): Training feature set
            y_train (pd.Series): Training label set
            X_test (pd.DataFrame): Test feature set
            y_test (pd.Series): Test label set
            tmpdirname: path to the temporary directory

        Returns:

        """
        learning_rate = trial.suggest_loguniform("learning_rate", 0.00001, 0.5)
        num_round = trial.suggest_loguniform("num_round", 1, 1000)
        alpha = trial.suggest_uniform("alpha", 0, 1)
        lambda_ = trial.suggest_loguniform("lambda", 0.0001, 10_000)
        subsample = trial.suggest_uniform("subsample", 0, 1)
        max_depth = trial.suggest_int("max_depth", 1, 8)
        scale_pos_weight = trial.suggest_loguniform("scale_pos_weight", 0.0001, 10_000)
        gamma = trial.suggest_loguniform("gamma", 0.0001, 100_000)
        min_child_weight = trial.suggest_loguniform("min_child_weight", 0.0001, 100_000)
        max_delta_step = trial.suggest_loguniform("min_child_weight", 0.0001, 10)

        # objective_ = 'binary:softmax' if unknown else 'binary:logistic'

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': learning_rate,
            'num_round': int(num_round),
            'alpha': alpha,
            'lambda': lambda_,
            'subsample': subsample,
            'max_depth': max_depth,
            'scale_pos_weight': scale_pos_weight,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'max_delta_step': max_delta_step
        }

        clf = XGBClassifier(**params)

        clf.fit(X_train, y_train)

        clf, cols = ModelTrainerImplementation.eval_shaps(clf, X_train, y_train, X_test, y_test, False)

        auc = ModelTrainerImplementation.score_callback(clf, X_test[cols], y_test)['f_1']

        tmp_path = os.path.join(tmpdirname, f'model{trial.number}.pkl')
        write_pickle(clf, tmp_path)

        trial.set_user_attr('cols', cols)

        return auc

    @staticmethod
    def tune_hps(Xtrain, ytrain, dval, yval, experiment_name):
        storage_name = f"sqlite:///artifacts/trials/{experiment_name}_xgb.db"
        study = optuna.create_study(direction="maximize", storage=storage_name)  # let it be fscore-1 and auc

        with tempfile.TemporaryDirectory() as tmpdirname:
            study.optimize(
                lambda trial: ModelTrainerImplementation.objective(trial, Xtrain, ytrain, dval, yval, tmpdirname),
                n_trials=50)

            print(experiment_name, tmpdirname)
            tmp_path = os.path.join(tmpdirname, f'model{study.best_trial.number}.pkl')
            model = load_pickle(tmp_path)

        return study.best_params, model, study.best_trial.user_attrs['cols']

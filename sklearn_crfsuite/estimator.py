# -*- coding: utf-8 -*-
from __future__ import absolute_import

from six.moves import zip
from tqdm import tqdm
import pycrfsuite
from sklearn.metrics import accuracy_score

from sklearn_crfsuite._fileresource import FileResource
from sklearn_crfsuite.utils import flatten
from sklearn_crfsuite.trainer import LinePerIterationTrainer
from sklearn_crfsuite.compat import BaseEstimator


class CRF(BaseEstimator):
    """
    python-crfsuite wrapper with interface siimlar to scikit-learn.
    It allows to use a familiar fit/predict interface and scikit-learn
    model selection utilities (cross-validation, hyperparameter optimization).

    Unlike pycrfsuite.Trainer / pycrfsuite.Tagger this object is picklable;
    on-disk files are managed automatically.

    Parameters
    ----------
    algorithm : str, optional (default='lbfgs')
        Training algorithm. Allowed values:

        * ``'lbfgs'`` - Gradient descent using the L-BFGS method
        * ``'l2sgd'`` - Stochastic Gradient Descent with L2 regularization term
        * ``'ap'`` - Averaged Perceptron
        * ``'pa'`` - Passive Aggressive (PA)
        * ``'arow'`` - Adaptive Regularization Of Weight Vector (AROW)

    min_freq : float, optional (default=0)
        Cut-off threshold for occurrence
        frequency of a feature. CRFsuite will ignore features whose
        frequencies of occurrences in the training data are no greater
        than `min_freq`. The default is no cut-off.

    all_possible_states : bool, optional (default=False)
        Specify whether CRFsuite generates state features that do not even
        occur in the training data (i.e., negative state features).
        When True, CRFsuite generates state features that associate all of
        possible combinations between attributes and labels.

        Suppose that the numbers of attributes and labels are A and L
        respectively, this function will generate (A * L) features.
        Enabling this function may improve the labeling accuracy because
        the CRF model can learn the condition where an item is not predicted
        to its reference label. However, this function may also increase
        the number of features and slow down the training process
        drastically. This function is disabled by default.

    all_possible_transitions : bool, optional (default=False)
        Specify whether CRFsuite generates transition features that
        do not even occur in the training data (i.e., negative transition
        features). When True, CRFsuite generates transition features that
        associate all of possible label pairs. Suppose that the number
        of labels in the training data is L, this function will
        generate (L * L) transition features.
        This function is disabled by default.

    c1 : float, optional (default=0)
        The coefficient for L1 regularization.
        If a non-zero value is specified, CRFsuite switches to the
        Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method.
        The default value is zero (no L1 regularization).

        Supported training algorithms: lbfgs

    c2 : float, optional (default=1.0)
        The coefficient for L2 regularization.

        Supported training algorithms: l2sgd, lbfgs

    max_iterations : int, optional (default=None)
        The maximum number of iterations for optimization algorithms.
        Default value depends on training algorithm:

        * lbfgs - unlimited;
        * l2sgd - 1000;
        * ap - 100;
        * pa - 100;
        * arow - 100.

    num_memories : int, optional (default=6)
        The number of limited memories for approximating the inverse hessian
        matrix.

        Supported training algorithms: lbfgs

    epsilon : float, optional (default=1e-5)
        The epsilon parameter that determines the condition of convergence.

        Supported training algorithms: ap, arow, lbfgs, pa

    period : int, optional (default=10)
        The duration of iterations to test the stopping criterion.

        Supported training algorithms: l2sgd, lbfgs

    delta : float, optional (default=1e-5)
        The threshold for the stopping criterion; an iteration stops
        when the improvement of the log likelihood over the last
        `period` iterations is no greater than this threshold.

        Supported training algorithms: l2sgd, lbfgs

    linesearch : str, optional (default='MoreThuente')
        The line search algorithm used in L-BFGS updates. Allowed values:

        * ``'MoreThuente'`` - More and Thuente's method;
        * ``'Backtracking'`` - backtracking method with regular Wolfe condition;
        * ``'StrongBacktracking'`` -  backtracking method with strong Wolfe
          condition.

        Supported training algorithms: lbfgs

    max_linesearch : int, optional (default=20)
        The maximum number of trials for the line search algorithm.

        Supported training algorithms: lbfgs

    calibration_eta : float, optional (default=0.1)
        The initial value of learning rate (eta) used for calibration.

        Supported training algorithms: l2sgd

    calibration_rate : float, optional (default=2.0)
        The rate of increase/decrease of learning rate for calibration.

        Supported training algorithms: l2sgd

    calibration_samples : int, optional (default=1000)
        The number of instances used for calibration.
        The calibration routine randomly chooses instances no larger
        than `calibration_samples`.

        Supported training algorithms: l2sgd

    calibration_candidates : int, optional (default=10)
        The number of candidates of learning rate.
        The calibration routine terminates after finding
        `calibration_samples` candidates of learning rates
        that can increase log-likelihood.

        Supported training algorithms: l2sgd

    calibration_max_trials : int, optional (default=20)
        The maximum number of trials of learning rates for calibration.
        The calibration routine terminates after trying
        `calibration_max_trials` candidate values of learning rates.

        Supported training algorithms: l2sgd

    pa_type : int, optional (default=1)
        The strategy for updating feature weights. Allowed values:

        * 0 - PA without slack variables;
        * 1 - PA type I;
        * 2 - PA type II.

        Supported training algorithms: pa

    c : float, optional (default=1)
        Aggressiveness parameter (used only for PA-I and PA-II).
        This parameter controls the influence of the slack term on the
        objective function.

        Supported training algorithms: pa

    error_sensitive : bool, optional (default=True)
        If this parameter is True, the optimization routine includes
        into the objective function the square root of the number of
        incorrect labels predicted by the model.

        Supported training algorithms: pa

    averaging : bool, optional (default=True)
        If this parameter is True, the optimization routine computes
        the average of feature weights at all updates in the training
        process (similarly to Averaged Perceptron).

        Supported training algorithms: pa

    variance : float, optional (default=1)
        The initial variance of every feature weight.
        The algorithm initialize a vector of feature weights as
        a multivariate Gaussian distribution with mean 0
        and variance `variance`.

        Supported training algorithms: arow

    gamma : float, optional (default=1)
        The tradeoff between loss function and changes of feature weights.

        Supported training algorithms: arow

    verbose : bool, optional (default=False)
        Enable trainer verbose mode.

    model_filename : str, optional (default=None)
        A path to an existing CRFSuite model.
        This parameter allows to load and use existing crfsuite models.

        By default, model files are created automatically and saved
        in temporary locations; the preferred way to save/load CRF models
        is to use pickle (or its alternatives like joblib).

    Attributes
    ----------
    tagger : pycrfsuite.Tagger
        python-crfsuite Tagger instance.

    """
    def __init__(self,
                 algorithm=None,

                 min_freq=None,
                 all_possible_states=None,
                 all_possible_transitions=None,
                 c1=None,
                 c2=None,
                 max_iterations=None,
                 num_memories=None,
                 epsilon=None,
                 period=None,
                 delta=None,
                 linesearch=None,
                 max_linesearch=None,
                 calibration_eta=None,
                 calibration_rate=None,
                 calibration_samples=None,
                 calibration_candidates=None,
                 calibration_max_trials=None,
                 pa_type=None,
                 c=None,
                 error_sensitive=None,
                 averaging=None,
                 variance=None,
                 gamma=None,

                 verbose=False,
                 model_filename=None,
                 keep_tempfiles=False,
                 trainer_cls=None):

        self.algorithm = algorithm
        self.min_freq = min_freq
        self.all_possible_states = all_possible_states
        self.all_possible_transitions = all_possible_transitions
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.num_memories = num_memories
        self.epsilon = epsilon
        self.period = period
        self.delta = delta
        self.linesearch = linesearch
        self.max_linesearch = max_linesearch
        self.calibration_eta = calibration_eta
        self.calibration_rate = calibration_rate
        self.calibration_samples = calibration_samples
        self.calibration_candidates = calibration_candidates
        self.calibration_max_trials = calibration_max_trials
        self.pa_type = pa_type
        self.c = c
        self.error_sensitive = error_sensitive
        self.averaging = averaging
        self.variance = variance
        self.gamma = gamma

        self.modelfile = FileResource(
            filename=model_filename,
            keep_tempfiles=keep_tempfiles,
            suffix=".crfsuite",
            prefix="model"
        )
        self.verbose = verbose
        self._tagger = None
        self.trainer_cls = trainer_cls

        self.training_log_ = None

    def fit(self, X, y, X_dev=None, y_dev=None):
        """
        Train a model.

        Parameters
        ----------
        X : list of lists of dicts
            Feature dicts for several documents (in a python-crfsuite format).

        y : list of lists of strings
            Labels for several documents.

        X_dev : (optional) list of lists of dicts
            Feature dicts used for testing.

        y_dev : (optional) list of lists of strings
            Labels corresponding to X_dev.
        """
        if (X_dev is None and y_dev is not None) or (X_dev is not None and y_dev is None):
            raise ValueError("Pass both X_dev and y_dev to use the holdout data")

        if self._tagger is not None:
            self._tagger.close()
            self._tagger = None
        self.modelfile.refresh()

        trainer = self._get_trainer()
        train_data = zip(X, y)

        if self.verbose:
            train_data = tqdm(train_data, "loading training data to CRFsuite", len(X), leave=True)

        for xseq, yseq in train_data:
            trainer.append(xseq, yseq)

        if self.verbose:
            print("")

        if X_dev is not None:
            test_data = zip(X_dev, y_dev)

            if self.verbose:
                test_data = tqdm(test_data, "loading dev data to CRFsuite", len(X_dev), leave=True)

            for xseq, yseq in test_data:
                trainer.append(xseq, yseq, 1)

            if self.verbose:
                print("")

        trainer.train(self.modelfile.name, holdout=-1 if X_dev is None else 1)
        self.training_log_ = trainer.logparser
        return self

    def predict(self, X):
        """
        Make a prediction.

        Parameters
        ----------
        X : list of lists of dicts
            feature dicts in python-crfsuite format

        Returns
        -------
        y : list of lists of strings
            predicted labels

        """
        return list(map(self.predict_single, X))

    def predict_single(self, xseq):
        """
        Make a prediction.

        Parameters
        ----------
        xseq : list of dicts
            feature dicts in python-crfsuite format

        Returns
        -------
        y : list of strings
            predicted labels

        """
        return self.tagger.tag(xseq)

    def predict_marginals(self, X):
        """
        Make a prediction.

        Parameters
        ----------
        X : list of lists of dicts
            feature dicts in python-crfsuite format

        Returns
        -------
        y : list of lists of dicts
            predicted probabilities for each label at each position

        """
        return list(map(self.predict_marginals_single, X))

    def predict_marginals_single(self, xseq):
        """
        Make a prediction.

        Parameters
        ----------
        xseq : list of dicts
            feature dicts in python-crfsuite format

        Returns
        -------
        y : list of dicts
            predicted probabilities for each label at each position

        """
        labels = self.tagger.labels()
        self.tagger.set(xseq)
        return [
            {label: self.tagger.marginal(label, i) for label in labels}
            for i in range(len(xseq))
        ]

    def score(self, X, y):
        """
        Return per-field accuracy score.
        """
        y_pred_flat = flatten(self.predict(X))
        y_true_flat = flatten(y)
        return accuracy_score(y_true_flat, y_pred_flat)

    @property
    def tagger(self):
        if self._tagger is None:
            if self.modelfile.name is None:
                raise Exception("Can't load model. Is the model trained?")

            tagger = pycrfsuite.Tagger()
            tagger.open(self.modelfile.name)
            self._tagger = tagger
        return self._tagger

    def _get_trainer(self):
        trainer_cls = self.trainer_cls or LinePerIterationTrainer
        params = {
            'feature.minfreq': self.min_freq,
            'feature.possible_states': self.all_possible_states,
            'feature.possible_transitions': self.all_possible_transitions,
            'c1': self.c1,
            'c2': self.c2,
            'max_iterations': self.max_iterations,
            'num_memories': self.num_memories,
            'epsilon': self.epsilon,
            'period': self.period,
            'delta': self.delta,
            'linesearch': self.linesearch,
            'max_linesearch': self.max_linesearch,
            'calibration.eta': self.calibration_eta,
            'calibration.rate': self.calibration_rate,
            'calibration.samples': self.calibration_samples,
            'calibration.candidates': self.calibration_candidates,
            'calibration.max_trials': self.calibration_max_trials,
            'type': self.pa_type,
            'c': self.c,
            'error_sensitive': self.error_sensitive,
            'averaging': self.averaging,
            'variance': self.variance,
            'gamma': self.gamma,
        }
        params = {k: v for k, v in params.items() if v is not None}
        return trainer_cls(
            algorithm=self.algorithm,
            params=params,
            verbose=self.verbose,
        )

    def __getstate__(self):
        dct = self.__dict__.copy()
        dct['_tagger'] = None
        return dct

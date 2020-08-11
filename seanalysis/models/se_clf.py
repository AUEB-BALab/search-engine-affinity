import numpy as np
from scipy import interp
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc, confusion_matrix, roc_curve
from seanalysis.models import Evaluation, Model
from seanalysis.algorithms.classifiers import classify
from seanalysis.utils import SEAnalysisException

BINARIZED_LABELS = {
    'roc': True,
    'cm': False
}


def roc_per_class(Y, pred, search_engines):
    """
    Calculate false positive rate and true positive rate per class.

    :param Y: Array with actual labels.
    :param pred: Array with predicted probabilities per class.
    :param search_engines: List of search_engines.

    Return: 1) Dictionary keyed by search engine which contains false positive
    rate per search_engine.
            2) Dictionary keyed by search engine which contains true positive
    rate per search_engine.
    """
    classes = Y.shape[1]
    fpr = {}
    tpr = {}
    for i in range(classes):
        fpr[search_engines[i]], tpr[search_engines[i]], _ = roc_curve(
            Y[:, i], pred[:, i])
    return fpr, tpr


class SEClassificationModel(Model):
    """
    This class represents a search engine classfier based on the results of
    search_engines.

    The goal of classification is to predict the search egine which generated
    a result (bag-of-words represenation) of a specific query.
    If search engines are similar, classifier should not be able to achieve
    a high score of classification.
    """
    def __init__(self, bows, queries, classifier='SVC', metric='roc', folds=3):
        self.bows = bows
        self.classifier = classifier
        self.metric = metric
        self.folds = folds
        self._queries = queries
        self._classes = [bow.se for bow in self.bows]
        self.X, self.Y = None, None
        self.results = None

    def construct(self):
        """
        Constructs the train data for the search engine classification problem.

        The construction of the train set is based on the metric which is going
        to be used for the evaluation of model.

        For example, Receiver Object Characteristic metric requires the
        binarizing of class labels.
        """
        try:
            self.construct_train_set(BINARIZED_LABELS[self.metric])
        except KeyError:
            raise SEAnalysisException(
                'Unsupported metric: %s' % repr(self.metric))

    def evaluate(self):
        """
        Evaluate the search engine classification model based on the specified
        metric, e.g. ROC, confusion matrix, etc.

        The evaluation is done via cross validation and for each iteration of
        cross validation process, the requested metric is computed for the
        returned score of classifier.
        """
        metrics = {
            'roc': self.roc_curve_analysis,
            'cm': self.confusion_matrix_analysis
        }
        try:
            metrics[self.metric]()
        except KeyError:
            raise SEAnalysisException(
                'Unsupported metric: %s' % repr(self.metric))

    def plot(self, visual, query_category):
        """
        Plot the results of analysis based on the metric which was used to
        evaluate model.

        :param visual: Visualization object used to plot the results.
        :param query_category: Category of queries used as identifier of
        the produced diagram.
        """
        if self.metric == 'roc':
            labels = [result.labels for result in self.results]
            metrics = [result.metrics for result in self.results]
            visual.plot_roc(metrics, labels, query_category)
        elif self.metric == 'cm':
            visual.plot_heatmap(
                self.results.metrics, self.results.labels, query_category)

    def construct_train_set(self, binarize_labels):
        """
        Construct classification model.

        Given the bag-of-words representation of a give query, the predicted
        value is search engine which generated this result.

        :param binarize_labels: True if labels should be binarized; False
        otherwise.
        """
        train_set = []
        Y = []
        for bow in self.bows:
            se_train = bow.matrix
            query_column = np.array(
                [self._queries.index(q) for q in self._queries]).reshape(
                    se_train.shape[0], 1)
            se_column = [self._classes.index(bow.se)] *\
                se_train.shape[0]
            train_set += [np.append(se_train, query_column, 1)]
            Y += se_column
        if binarize_labels:
            labels = label_binarize(np.array(Y), classes=range(
                len(self._classes)))
            if len(self._classes) == 2:
                labels = np.hstack((labels, 1 - labels))
        else:
            labels = np.vstack(Y).ravel()
        self.X, self.Y = np.vstack(train_set), labels

    def roc_curve_analysis(self):
        """
        Evaluation of classication model using roc metric via cross validation.

        This method calculates true and false positive rate per class for
        every iteration of cross validation loop. At the end of cross
        validation process the mean true and false positive rate is calculated
        per class.
        """
        mean_tpr = {se: 0.0 for se in self._classes}
        mean_fpr = {se: np.linspace(0, 1, 100) for se in self._classes}
        self.results = []
        kf = KFold(n_splits=self.folds, shuffle=True,
                   random_state=None)
        for train_index, test_index in kf.split(self.X):
            pred = classify(self.X[train_index], self.Y[train_index, :],
                            self.X[test_index], classifier=self.classifier,
                            oneVSrest=True)
            fpr, tpr = roc_per_class(
                self.Y[test_index, :], pred, self._classes)
            for se in self._classes:
                mean_tpr[se] += interp(mean_fpr[se], fpr[se], tpr[se])
                mean_tpr[se][0] = 0.0
        for se in self._classes:
            mean_tpr[se] /= self.folds
            mean_tpr[se][-1] = 1.0
            mean_auc = auc(mean_fpr[se], mean_tpr[se])
            self.results.append(Evaluation(
                se, (mean_fpr[se], mean_tpr[se], mean_auc), 'se'))

    def confusion_matrix_analysis(self):
        """
        This function calculates the confusion matrix using cross validation.
        """
        N = len(self._classes)
        total_cm = np.zeros((N, N), dtype=float)
        kf = KFold(n_splits=self.folds, shuffle=True,
                   random_state=None)
        for train_index, test_index in kf.split(self.X):
            pred = classify(self.X[train_index], self.Y[train_index],
                            self.X[test_index], classifier=self.classifier,
                            oneVSrest=False, prob=False)
            cm = confusion_matrix(self.Y[test_index], pred)
            total_cm += cm
        norm_cm = total_cm.astype('float') / total_cm.sum(
            axis=1)[:, np.newaxis]
        self.results = Evaluation(self._classes, norm_cm, 'se')

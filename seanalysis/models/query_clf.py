# Copyright (c) 2016-2020 AUEB BaLab
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import itertools
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
import seanalysis.algorithms.classifiers as clfs
from seanalysis.models import Evaluation, Model
from seanalysis.utils import SEAnalysisException


class QueryClassificationModel(Model):
    """
    This model is based on a classification problem, having queries as labels.

    Given the bag of words representation of results, this model aims to
    predict the queries associated with the produced results.

    The evaluation of this model is done via cross learn compare technique
    which uses the results of one search engines as train data and the results
    of another search engine as test data. It is expected that if two search
    engines are similar, then the classifier trained by the results of the
    search engine A can achieve high score predicting the results of search
    engine B.

    Receiver object characteristic metric is used for the evaluation.
    """
    def __init__(self, bows, queries, classifier='SVC', evaluation=None):
        self.bows = bows
        self.classifier = classifier
        self.evaluation = evaluation
        self._queries = queries
        self.results = None

    def construct(self):
        """
        `construct` method is specified on `models.Model` abstract method.
        """
        pass

    def evaluate(self):
        """
        Evaluate query classification model based on the specifier evaluation
        method.
        """
        evaluation_methods = {
            'clc': self.cross_learn_compare
        }
        try:
            evaluation_methods[self.evaluation]()
        except KeyError:
            raise SEAnalysisException(
                'Unsupported evaluation method: %s' % repr(self.evaluation))

    def plot(self, visual, query_category):
        """
        Plot the results of the analysis based on the evaluation technique.

        :param visual: Visualization object used to plot the results.
        :param query_category: Category of queries used as identifier of
        the produced diagram.
        """
        plot_methods = {
            'clc': self.plot_cross_learn_compare
        }
        try:
            plot_methods[self.evaluation](visual, query_category)
        except KeyError:
            raise SEAnalysisException(
                'Unsupported evaluation method: %s' % repr(self.evaluation))

    def cross_learn_compare(self):
        """
        This method uses the cross learn compare technique which for every
        permutation of search_engines results, it trains a classifier with
        the results of search engine A and tests it with the results of
        search engine B.

        It expected that if two search engines are similar, then classifier
        can achieve high score on predicting results of search engine B based
        on results of search engine A.

        It uses the Receiver Object Characterstic metric to evaluate every
        instance.
        """
        # TODO In case of n search engines try to use the results of n-1 search
        # engines as train data in order to predict the results of the remnant
        # search engine.
        self.results = []
        for a, b in itertools.permutations(self.bows, 2):
            Y = label_binarize(self._queries, classes=list(set(self._queries)))
            pred = clfs.classify(b.matrix, Y, a.matrix,
                                 classifier=self.classifier, oneVSrest=True)
            fpr, tpr, _ = roc_curve(Y.ravel(), pred.ravel())
            self.results.append(Evaluation(
                b.se + '->' + a.se, (fpr, tpr, auc(fpr, tpr)), 'clc'))

    def plot_cross_learn_compare(self, visual, query_category):
        """
        Plot the results of the cross learn analysis.

        This method plots ROC curve for every classifier instance.
        For example, ROC curve of the predictions of search engine B results
        based on the results of search engine B.

        :param visual: Visualization object used to plot the results.
        :param query_category: Category of queries used as identifier of
        the produced diagram.
        """
        roc_metrics = [result.metrics for result in self. results]
        labels = [result.labels for result in self.results]
        visual.plot_roc(roc_metrics, labels, query_category)

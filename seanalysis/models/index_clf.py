import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from seanalysis.models import Evaluation, Model
from seanalysis.algorithms.classifiers import classify


class IndexClassificationModel(Model):
    """
    This class is responsible for the Index Classification problem.

    This problem aims to predict successfully the index of a given result of
    query produced by a specific search engine. Train set of this model
    includes the bag of words representation of the queries results of
    search engines. It does not include any information about which search
    engine produced this result. The classes of this model are the indexes
    of the result.

    This problem is a complementary of the search engine classification
    problem. We first should prove that two search engines are similar from
    the poor classifier's performance on the aforementioned classification
    problem. If classifier on the index classification problem does well,
    then we can infer that these search engines also tend to produce
    results on similar indexes.

    However, if classifier on search engine classification problem achieved
    high performance then of course, we can't infer that two search engines
    produce results on the same index, although classifier on the index
    classification problem does well.
    """
    # TODO Support more evaluation metrics.
    def __init__(self, bows, queries, indexes, classifier='LOGREG',
                 metric=None, folds=3):
        self.bows = bows
        self._queries = queries
        self.classifier = classifier
        self.folds = folds
        self.indexes, self._classes = indexes, set(indexes)

    def construct(self):
        """
        Constructs the train set the classes of this classification problem.

        Train set contains the bag of words representation of the results of
        all queries that specified search engines produced without any
        information of which search engine actually returned a specific
        result.

        Classes of this model are the indexes of the results.
        """
        train_set = []
        Y = []
        query_set = list(set(self._queries))
        for bow in self.bows:
            se_train = bow.matrix
            query_column = np.array(
                [query_set.index(q) for q in self._queries]).reshape(
                    se_train.shape[0], 1)
            train_set += [np.append(se_train, query_column, 1)]
            Y += self.indexes
        self.X, self.Y = np.vstack(train_set), np.vstack(Y).ravel()

    def evaluate(self):
        """
        This method evaluates the performance of the classifier of the current
        problem.

        It uses cross validation along with the specified classifer and the
        number of folds. To evaluate the performance of the classifier
        confusion matrix is used.
        """
        N = len(self._classes)
        total_cm = np.zeros((N, N), dtype=float)
        kf = KFold(self.X.shape[0], n_folds=self.folds, shuffle=True,
                   random_state=None)
        for train_index, test_index in kf:
            pred = classify(self.X[train_index], self.Y[train_index],
                            self.X[test_index], classifier=self.classifier,
                            oneVSrest=False, prob=False)
            cm = confusion_matrix(self.Y[test_index], pred)
            total_cm += cm
        norm_cm = total_cm.astype('float') / total_cm.sum(
            axis=1)[:, np.newaxis]
        self.results = Evaluation(self._classes, norm_cm, 'se')

    def plot(self, visual, query_category):
        """
        This method is responsible for plotting the results of the evaluation.

        :param visual: Object responsible for plotting reults of classifier.
        :param query_category: Categories of queries which participated to this
        problem. It's like an identifier of the results.
        """
        visual.plot_heatmap(
            self.results.metrics, self.results.labels, query_category)

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

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


CLASSIFIERS = {
    'SVC': SVC(probability=True, kernel='linear'),
    'LOGREG': LogisticRegression()
}


def classify(train_set, train_labels, test_set, classifier, oneVSrest=False,
             prob=True):
    """
    Predict label with linear SVC.

    :param train_et: Set of data to train the classifier.
    :param train_label: Labels of train set.
    :param tet_set: Set of data to test the classifier.
    :param classifier: Classifier name which is going to be used to train data.
    :param oneVSrest: True if a One vs Rest Strategy is going to be used on
    the classification process.
    :param prob: True if estimated probabilities for each class are returned;
    False if classifier predictions are returned

    :return: Predicted labels of classifier or estimated probabilities
    for every class.
    """
    clf = OneVsRestClassifier(CLASSIFIERS[classifier]) if oneVSrest\
        else CLASSIFIERS[classifier]
    clf.fit(train_set, train_labels)
    return clf.predict_proba(test_set) if prob else clf.predict(test_set)

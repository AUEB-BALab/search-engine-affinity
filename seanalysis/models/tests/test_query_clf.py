import unittest
import mock
import numpy as np
import seanalysis.models.query_clf as q
from seanalysis.utils import SEAnalysisException

class TestQueryClassificationModel(unittest.TestCase):
    def setUp(self):
        self.mockModel = mock.MagicMock(
            evaluation='clc',
            evaluate=q.QueryClassificationModel.__dict__['evaluate'],
            plot=q.QueryClassificationModel.__dict__['plot']
        )

    def test_evaluate(self):
        try:
            self.mockModel.evaluate(self.mockModel)
        except SEAnalysisException:
            self.fail('Unexpected exception')
        self.assertEqual(self.mockModel.cross_learn_compare.call_count, 1)
        self.mockModel.evaluation = ''
        self.assertRaises(SEAnalysisException, self.mockModel.evaluate,
                          self.mockModel)
        self.assertEqual(self.mockModel.cross_learn_compare.call_count, 1)

    def test_plot(self):
        try:
            self.mockModel.plot(self.mockModel, None, '')
        except SEAnalysisException:
            self.fail('Unexpected exception')
        self.assertEqual(self.mockModel.plot_cross_learn_compare.call_count, 1)
        self.mockModel.evaluation = ''
        self.assertRaises(
            SEAnalysisException, self.mockModel.plot, self.mockModel, None, '')
        self.assertEqual(self.mockModel.plot_cross_learn_compare.call_count, 1)

    def test_cross_learn_compare(self):
        self.mockModel.cross_learn_compare = q.QueryClassificationModel\
                .__dict__['cross_learn_compare']
        bow1 = mock.MagicMock(se='a', matrix=np.array(
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]]))
        bow2 = mock.MagicMock(se='b', matrix=np.array(
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]]))
        self.mockModel.bows = [bow1, bow2]
        self.mockModel._queries = ['a', 'b', 'c']
        self.mockModel.classifier = 'SVC'
        self.mockModel.cross_learn_compare(self.mockModel)
        self.assertEqual(len(self.mockModel.results), 2)
        evaluation = self.mockModel.results[0]
        self.assertEqual(evaluation.labels, 'b->a')
        _, _, auc = evaluation.metrics
        self.assertEqual(auc, 0.5)
        self.assertEqual(evaluation.type, 'clc')

        evaluation = self.mockModel.results[1]
        self.assertEqual(evaluation.labels, 'a->b')
        _, _, auc = evaluation.metrics
        self.assertEqual(auc, 0.5)
        self.assertEqual(evaluation.type, 'clc')

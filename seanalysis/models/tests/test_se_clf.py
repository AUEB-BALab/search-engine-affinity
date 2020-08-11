import unittest
import mock
import seanalysis.models.se_clf as se
from seanalysis.utils import SEAnalysisException


class TestSEClassificationModel(unittest.TestCase):
    def setUp(self):
        self.mockModel = mock.MagicMock(
            metric='roc',
            construct=se.SEClassificationModel.__dict__['construct'],
            evaluate=se.SEClassificationModel.__dict__['evaluate'],
            plot=se.SEClassificationModel.__dict__['plot']
        )

    def test_construct(self):
        try:
            self.mockModel.construct(self.mockModel)
        except SEAnalysisException:
            self.fail('Unexpected SEAnalysisException')
            self.mockModel.construct_train_set.assert_called_once_with(
                se.BINARIZED_LABELS[self.mockModel.metric])
        self.mockModel.metric = ''
        self.assertRaises(SEAnalysisException, self.mockModel.construct,
                          self.mockModel)
        self.assertEqual(self.mockModel.construct_train_set.call_count, 1)

    def test_evaluate(self):
        try:
            self.mockModel.evaluate(self.mockModel)
        except SEAnalysisException:
            self.fail('Unexpected exception')
        self.assertEqual(self.mockModel.roc_curve_analysis.call_count, 1)
        self.mockModel.metric = ''
        self.assertRaises(SEAnalysisException, self.mockModel.evaluate,
                          self.mockModel)
        self.assertEqual(self.mockModel.roc_curve_analysis.call_count, 1)

    def test_plot(self):
        mock_visual = mock.MagicMock()
        self.mockModel.plot(self.mockModel, mock_visual, '')
        self.assertEqual(mock_visual.plot_roc.call_count, 1)
        self.assertEqual(mock_visual.plot_confusion_matrix.call_count, 0)
        self.mockModel.metric = 'cm'
        self.mockModel.plot(self.mockModel, mock_visual, '')
        self.assertEqual(mock_visual.plot_roc.call_count, 1)
        self.assertEqual(mock_visual.plot_confusion_matrix.call_count, 1)

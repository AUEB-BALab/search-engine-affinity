import unittest
import mock
from seanalysis.controller import controller as ctrl
from seanalysis.utils import SEAnalysisException

SUPPORTED_MODELS = {
    "model1": mock.MagicMock(),
    "model2": mock.MagicMock()
}

SUPPORTED_METHODS = ["method1", "method2"]

SUPPORTED_MODELS_PER_METHOD = {
    "method1": ["model1"],
    "method2": ['model2']
}

CONFIGURATION_SCHEMA = {
    "method1": {
        'model1': {
            "a": int,
            'b': int
        }
    }
}

ctrl.SUPPORTED_MODELS = SUPPORTED_MODELS
ctrl.SUPPORTED_METHODS = SUPPORTED_METHODS
ctrl.SUPPORTED_MODELS_PER_METHOD = SUPPORTED_MODELS_PER_METHOD
ctrl.CONFIGURATION_SCHEMA = CONFIGURATION_SCHEMA


class TestController(unittest.TestCase):

    def test_validate(self):
        controller = ctrl.Controller('method1', 'model1', [])
        try:
            controller.validate()
        except SEAnalysisException:
            self.fail('Unexpected exception')

        controller.method = 'wrong method'
        self.assertRaises(SEAnalysisException, controller.validate)

        controller = ctrl.Controller('method1', 'wrong model', [])
        self.assertRaises(SEAnalysisException, controller.validate)

        controller.model = 'model2'
        self.assertRaises(SEAnalysisException, controller.validate)

    def test_parse_config(self):
        controller = ctrl.Controller('method1', 'model1', [])
        self.assertRaises(SEAnalysisException, controller.parse_config)

        controller.config = ['invalid config']
        self.assertRaises(SEAnalysisException, controller.parse_config)

        controller.config = ['invalid_key=value']
        self.assertRaises(SEAnalysisException, controller.parse_config)

        controller.config = ['a=invalid_value']
        self.assertRaises(SEAnalysisException, controller.parse_config)

        controller.config = ['a=1']
        self.assertRaises(SEAnalysisException, controller.parse_config)

        controller.config = ['a=1', 'b=2', 'c=3']
        self.assertRaises(SEAnalysisException, controller.parse_config)

        controller.config = ['a=1', 'b=2']
        config = controller.parse_config()
        self.assertEqual(config, {'a': 1, 'b': 2})

    @mock.patch.object(ctrl.Controller, 'validate')
    @mock.patch.object(ctrl.Controller, 'parse_config')
    @mock.patch('seanalysis.controller.controller.BagOfWords.build_bows')
    @mock.patch('seanalysis.controller.controller.BagOfWords.get_queries')
    @mock.patch('seanalysis.controller.controller.Visualization.show')
    def test_analyze(self, mock_visual, mock_bow, mock_queries, mock_parse,
                     mock_validate):
        controller = ctrl.Controller('method1', 'model1', [])
        mock_bow.return_value = []
        mock_queries.return_value = []
        controller.analyze({'a': {}, 'b': {}}, None, True)
        self.assertEqual(mock_validate.call_count, 1)
        self.assertEqual(mock_parse.call_count, 1)
        self.assertEqual(mock_bow.call_count, 2)
        self.assertEqual(mock_queries.call_count, 2)
        self.assertEqual(mock_visual.call_count, 1)

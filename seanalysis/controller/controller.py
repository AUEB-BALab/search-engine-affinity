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

from seanalysis.models.query_clf import QueryClassificationModel
from seanalysis.models.index_clf import IndexClassificationModel
from seanalysis.models.lda import LDA
from seanalysis.models.se_clf import SEClassificationModel
from seanalysis.models.tensor import TensorCompare
from seanalysis.algorithms.bag_of_words import BagOfWords
from seanalysis.drawing.visualization import Visualization
from seanalysis.utils import SEAnalysisException


SUPPORTED_MODELS = {
    "tensor": TensorCompare,
    "lda": LDA,
    "query": QueryClassificationModel,
    "se": SEClassificationModel
}

SUPPORTED_METHODS = ["cmp", "clf"]

SUPPORTED_MODELS_PER_METHOD = {
    "cmp": ["tensor", "lda"],
    "clf": ["query", "se"]
}

CONFIGURATION_SCHEMA = {
    "cmp": {
        'lda': {
            "components": int
        },
        'tensor': {
            'components': int
        }
    },
    "clf": {
        'query': {
            "classifier": str,
            "evaluation": str
        },
        'se': {
            'classifier': str,
            'metric': str,
            'folds': int
        }
    }
}


class Controller(object):
    """
    This class controls the analysis of the similarity of search engines based
    on the specified method, model and configuration.

    This controller class is responsible for the following processes.
        - First of all, there is a validation of the given method, model and
        the configuaration.
        - After validation, chooses the specified model to construct and fit
        the train data and plot the results of the analysis.
    """
    def __init__(self, method, model, config):
        self.method = method
        self.model = model
        self.config = config

    def validate(self):
        """
        This method validates the given method and model if they are supported
        by the current controller. It raises an error if this is not the case.

        :raises: Exception if given method and model are invalid.
        """
        if self.method not in SUPPORTED_METHODS:
            raise SEAnalysisException(
                'Method %s not supported' % repr(self.method))

        if self.model not in SUPPORTED_MODELS:
            raise SEAnalysisException(
                'Model %s not supported' % repr(self.model))

        if self.model not in SUPPORTED_MODELS_PER_METHOD[self.method]:
            raise SEAnalysisException(
                'Model %s is not supported by method %s' % (
                    repr(self.model), repr(self.method)))

    def parse_config(self):
        """
        This method checks the if the given configuration is valid based on the
        model and its schema.

        For example, given configuration should be compatible with the
        specified fields and type of field values.

        Then, constructs a dictionary based on the key-value pair.

        :raises: Exception if the given configuration is not valid
        """
        required_config = CONFIGURATION_SCHEMA[self.method][self.model].copy()
        parsed_config = {}
        if not self.config:
            raise SEAnalysisException(
                'Not any config value for method %s' % repr(self.method))

        for conf in self.config:
            pair = tuple(conf.split('='))
            if len(pair) != 2:
                raise SEAnalysisException('Invalid config %s' % repr(conf))

            key, value = pair
            if key not in required_config:
                raise SEAnalysisException('Invalid config %s for model %s' % (
                    repr(conf), repr(self.model)))
            value_type = required_config.pop(key)
            try:
                parsed_config[key] = value_type(value)
            except ValueError:
                raise SEAnalysisException(
                    'Invalid value type for field %s' % repr(key))

        if required_config:
            raise SEAnalysisException('Config (%s) for model %s is required' % (
                ', '.join(required_config.keys()), repr(self.model)))
        return parsed_config

    def analyze(self, data, index, N, merge):
        """
        This method triggers the analysis of data given as parameter.

        First of all, it validates the specified method model and
        configuration. Then, for every query category it takes the bag of
        words representation of query results and creates a new model
        to construct and fit these data, as well as plots the results
        of the analysis of model.

        :param data: Dictionary keyed by query category which contains the
        snippets for every query result of every search engine.
        :param N: Length of vocabulary. If `None` uses all terms for
        the vocabulary.
        :param merge: True to produce a single diagram for the analysis of
        all query categories; False otherwise.

        """
        self.validate()
        config = self.parse_config()
        models = [SUPPORTED_MODELS[self.model], IndexClassificationModel]\
            if index else [SUPPORTED_MODELS[self.model]]
        visual = Visualization(merge, len(data) * len(models))
        for query_category, snippets in data.items():
            bow = BagOfWords(snippets)
            bow_obj = bow.build_bows(N)
            for model_cls in models:
                model_obj = (
                    model_cls(bow_obj, bow.get_queries(), **config)
                    if model_cls is not IndexClassificationModel
                    else model_cls(
                        bow_obj, bow.get_queries(), bow.get_indexes(),
                        **config)
                )
                model_obj.construct()
                model_obj.evaluate()
                model_obj.plot(visual, query_category)
        visual.show()

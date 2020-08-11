from abc import ABCMeta, abstractmethod
from collections import namedtuple


Evaluation = namedtuple('Evaluation', ['labels', 'metrics', 'type'])


class Model(object):
    """
    This class defines models for the analysis of search engines.

    A model should construct and transorm data in the required format which
    is different for different models.
    Moreover, a model should fit and evaluate train data using Machine Learning
    algorithms and plotting the results of the analysis.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def construct(self):
        """ Constucts the train data to be used by model. """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Fit and evaluate train data based on the configuration of model.
        """
        pass

    @abstractmethod
    def plot(self, vsl, title):
        """
        Plot the results of the analysis.

        :param vsl: The obj responsible for the plotting of data.
        :param title: Title to identify the diagram of analysis.
        """
        pass

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

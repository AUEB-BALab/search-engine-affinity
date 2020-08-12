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

from __future__ import division
from itertools import chain
import numpy as np
from sktensor import ktensor, dtensor
from sktensor.core import khatrirao
from seanalysis.models import Model


class TensorCompare(Model):
    """
    Tensor compare model uses tensors to represent results and then using
    CP APR algorithm tries to decomposes initial tensor to components.
    """

    def __init__(self, bows, queries, components=20):
        self.X = None
        self.bows = bows
        self.components = components
        self._queries = queries
        self._search_engines = [bow.se for bow in bows]
        self.se_dist = None
        self.query_dist = None

    def construct(self):
        """
        This method creates a tensor with the following modes: (search engine,
        query, date, term).

        This tensor is constructed from the bag of words representation
        matrices; one for each search engine. These matrices have the following
        shape: (number of query X date, number of terms).

        The idea is to construct the required tensor from a tensor which has
        3 modes (query, date, term).

        :param snippets: Dictionary keyed by search engine that contains the
        concatenation of snippets for every result for every query.
        :param queries: Set of queries.

        :return: Tensor with the four modes (search engine, query, date, term)
        """
        # Get a the first value of dictionary.
        # Any other value could be used instead.
        query_index = {query: i for i, query in enumerate(
            set(self._queries))}

        query_columns = np.zeros((self.bows[0].matrix.shape[0], 1), dtype=int)
        arrays_3d = []
        se_bows = [bow.matrix for bow in self.bows]
        for i, bow in enumerate(se_bows):
            # Add a new column to the the bag of words representation matrix.
            # This new column refers to the query which this representation is
            # associated with. This is going to be used to split array in a
            # list of sub-arrays. Each sub array has the same query column and
            # obviously has as many rows as the number of days.
            bow = np.append(bow, query_columns, 1)
            for j, row in enumerate(bow):
                query = self._queries[j]
                row[-1] = query_index[query]
            bow = bow[bow[:, -1].argsort()]
            splits = np.split(bow, np.where(
                np.diff(np.array(bow[:, -1], dtype=int)))[0] + 1)
            # Adds a new 3d array with the following shape (query, term, date).
            # This list contains the 3d array for every search engine.
            arrays_3d.append(np.concatenate(splits).reshape(
                (len(splits),) + splits[0].shape))
            se_bows[i] = bow
        new_shape = (len(arrays_3d),) + arrays_3d[0].shape
        self.X = dtensor(
            np.concatenate(se_bows).reshape(new_shape)[:, :, :, :-1])

    def evaluate(self):
        """ This method fits created tensor using CP APR algorithm. """
        M = cp_apr(self.X, self.components)
        self.se_dist = M.U[0]
        self.query_dist = M.U[2]

    def plot(self, visual, query_category):
        """
        This method plots the contribution of every search engine to every
        cluster.

        If the search engines are similar, it is expected that they contribute
        to every cluster with a similar way.

        :param visual: Visualization obj responsible for the plotting of
        resutlts.
        :param query_category: Category of queries which were used for the
        analysis.
        """
        visual.draw_se_clusters(self.se_dist, query_category,
                                self._search_engines)


def cp_apr(X, r, M=None, outer_iter=1000, inner_iter=10, t=1e-4, k=0.01,
           k_tol=1e-10, e=1e-10):
    """
    Implementation of CP-APR algorithm.

    http://arxiv.org/pdf/1112.2414.pdf

    :param X: tensor to decompose.
    :param r: Number of R components.
    :param M: Initial guess of R-Component model.
    :param outer_iter: Number of maximum outer iterations.
    :param inner_iter: Number of maximum inner iterations.
    :param t: Convergence tolerance of KKT conditions.
    :param k: Inadmissible zero avoidance adjustment.
    :param k_tol: Tolerance of identifying a potentional inadmissible zero.
    :param e: Minimum divisor to prevent divide-by-zero.

    :returns: Decomposition of tensor to `N` components.
    """
    N = len(X.shape)
    if M is None:
        M = ktensor([np.random.rand(X.shape[i], r) for i in range(N)])
    phi = np.empty([N, ], dtype=object)

    for i in range(outer_iter):
        is_converged = True
        for n in range(N):
            S = np.zeros((X.shape[n], r))
            if i > 0:
                S[(phi[n] > 1) & (M.U[n] < k_tol)] = k
            b = np.dot((M.U[n] + S), np.diag(M.lmbda))
            pi = khatrirao(tuple(
                [M.U[i] for i in chain(range(n), range(n + 1, N))]),
                reverse=True).transpose()
            pi_transpose = pi.transpose()
            X_unfold = X.unfold(n)
            for j in range(inner_iter):
                phi[n] = np.dot(X_unfold / np.maximum(np.dot(b, pi), e),
                                pi_transpose)
                if np.amax(np.abs(np.ravel(np.minimum(b, 1 - phi[n])))) < t:
                    break

                is_converged = False
                b *= phi[n]
            M.lmbda = np.dot(np.ones(b.shape[0]).transpose(), b)
            M.U[n] = np.dot(b, np.linalg.inv(np.diag(M.lmbda)))

        if is_converged:
            break
    return M

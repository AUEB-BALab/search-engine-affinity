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

import math
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def gridspec_shape(N):
    root = math.sqrt(N)
    round_root = int(root)
    return (round_root, round_root) if root == round_root\
        else (round_root, round_root + 1)


class Visualization(object):
    """
    This class is responsible for plotting results of multiple analyses.
    """
    def __init__(self, merge, N, geometry=False):
        self.merge = merge
        self.fig_serial = 0
        x, y = (N / 10, N % 10) if geometry else gridspec_shape(N)
        self.gridspec = None
        self.subplot_serial = None
        if merge:
            self.gridspec = gridspec.GridSpec(x, y, wspace=0.1, hspace=0.0)
            self.subplot_serial = 0

    def next_figure(self):
        """ Specify the next figure or subplot. """
        if self.merge:
            self.subplot_serial += 1
        else:
            self.fig_serial += 1

    def draw_se_clusters(self, se_dist, analysis_identifier, search_engines):
        """
        Plot the contribtion of every search engine to every cluster specified
        by a component based model.

        :param se_dist: Contribution of every search engine to every cluster.
        :param analysis_identifier: Identifier for the current analysis.
        :param search_engines: List of search engines.
        """
        if se_dist.shape[0] == 2:
            self._se_activity_2d(se_dist, analysis_identifier, search_engines)
        else:
            self._se_activity_3d(se_dist, analysis_identifier, search_engines)
        self.next_figure()

    def _se_activity_3d(self, se_dist, analysis_identifier, search_engines):
        """
        Draw contribution of three search engines in 3D.

        :param se_dist: Contribution of every search engine to every cluster.
        :param analysis_identifier: Identifier for the current analysis.
        :param search_engines: List of search engines.
        """
        fig = plt.figure(self.fig_serial)
        grid = self.gridspec[self.subplot_serial] if self.merge else 111
        ax = fig.add_subplot(grid, projection='3d')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(0.0, 1.0)
        ax.set_xlabel(search_engines[0], fontsize=14)
        ax.set_ylabel(search_engines[1], fontsize=14)
        ax.set_zlabel(search_engines[2], fontsize=14)
        ax.set_title('Search engine clusters (%s)' % (analysis_identifier),
                     fontsize=14)
        ax.scatter(se_dist[0, :], se_dist[1, :],
                   se_dist[2, :], s=50, alpha=0.2)

    def _se_activity_2d(self, se_dist, analysis_identifier, search_engines):
        """
        Draw contribution of two search engines in 2D.

        :param se_dist: Contribution of every search engine to every cluster.
        :param analysis_identifier: Identifier for the current analysis.
        :param search_engines: List of search engines.
        """
        fig = plt.figure(self.fig_serial)
        if self.merge:
            ax = plt.Subplot(fig, self.gridspec[self.subplot_serial])
            fig.add_subplot(ax)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel(search_engines[0], fontsize=14)
        plt.ylabel(search_engines[1], fontsize=14)
        plt.title('Search engine clusters(%s)' % (analysis_identifier),
                  fontsize=14)
        plt.plot(se_dist[0, :], se_dist[1, :], 'o', ms=20, alpha=0.2)

    def draw_day_clusters(self, day_memberships, analysis_identifier):
        """
        Draw contribution of each day to the cluster.

        :param day_memberships: Component of decomposed tensor with the
        contribution of each day when results were collected.
        :param analysis_identifier: Category of query.
        """
        fig = plt.figure(self.fig_serial)
        if self.merge:
            ax = plt.Subplot(fig, self.gridspec[self.subplot_serial])
            fig.add_subplot(ax)
        days, R = day_memberships.shape
        for i in range(R):
            plt.plot(xrange(days), day_memberships[:, i])
        plt.axis([0, days - 1, 0, 1.0])
        plt.xlabel('Day')
        plt.ylabel('Activity of cluster')
        plt.title('Day clusters(%s)' % (analysis_identifier))

    def plot_roc(self, roc_metrics, labels, analysis_identifier):
        """
        Plot the given ROC curves.

        :param roc_metrics: List of tuples with the required metrics for
        plotting ROC curves, i.e. True positive rate, False positive rate.
        :param labels: List of labels for each ROC curve.
        :param analysis_identifier: Identifier for the analysis.
        """
        fig = plt.figure(self.fig_serial)
        if self.merge:
            ax = plt.Subplot(fig, self.gridspec[self.subplot_serial])
            fig.add_subplot(ax)
        for i, (fpr, tpr, auc) in enumerate(roc_metrics):
            plt.plot(fpr, tpr, label='{0} (area = {1:0.2f})'
                     ''.format(labels[i], auc))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(
            'Receiver Operating Characteristic curve (%s)' % (
                analysis_identifier), fontsize=14)
        plt.legend(loc="lower right")
        self.next_figure()

    def plot_heatmap(self, data, matrix_labels, analysis_identifier):
        """
        Plot a heatmap.

        :param data: A matrix with the data to be plotted.
        :param matrix_labels: List of matrix labels.
        :param analysis_identifier: Identifier for the analysis.
        """
        fig = plt.figure(self.fig_serial)
        if self.merge:
            ax = plt.Subplot(fig, self.gridspec[self.subplot_serial])
            fig.add_subplot(ax)
        ax = sns.heatmap(data, vmin=0, vmax=1, cmap='RdBu')
        ax.set_xlabel(matrix_labels[0], fontsize=14)
        ax.set_ylabel(matrix_labels[1], fontsize=14)
        plt.title(analysis_identifier, fontsize=14)
        self.next_figure()

    def plot_day_similarity(self, data, analysis_identifier, weight_a,
                            weight_b, weight_c):
        """
        Plot similarity of two search engines over time.

        :param data: A matrix with the data to be plotted.
        """
        fig = plt.figure(self.fig_serial)
        if self.merge:
            ax = plt.Subplot(fig, self.gridspec[self.subplot_serial])
            fig.add_subplot(ax)

        markers = cycle(('o', '>', 'v', 's', '<', '^'))
        for i in range(len(data[0, 0, :])):
            d = np.mean(data[:, :, i], axis=1)
            label = 'a=' + str(weight_a[i]) + ', ' + 'b=' + str(weight_b[i])\
                    + ', ' + 'c=' + str(weight_c[i])
            plt.plot(d, marker=markers.next(), label=label, alpha=0.6,
                     linewidth=2, ms=7)
        plt.ylim([0, 1.0])
        a = plt.legend(fancybox=True, loc='best', prop={'size': 10})
        a.get_frame().set_alpha(1)
        plt.ylabel('Similarity', fontsize=14)
        plt.xlabel('Day', fontsize=14)
        plt.title(analysis_identifier, fontsize=16)
        self.next_figure()

    def show(self):
        """ Show existed plots. """
        plt.show()

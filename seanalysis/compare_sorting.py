# coding=utf-8

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

import random
import string
from difflib import SequenceMatcher
import numpy as np
from jellyfish import levenshtein_distance, damerau_levenshtein_distance, \
    hamming_distance, jaro_distance, jaro_winkler
from seanalysis import metrics


def normalize_levenshtein_distance(se1, se2):
    return 1-float(levenshtein_distance(se1, se2))/len(se1)


def normalize_damerau_levenshtein_distance(se1, se2):
    return 1-float(damerau_levenshtein_distance(se1, se2))/len(se1)


def normalize_hamming_distance(se1, se2):
    return 1-float(hamming_distance(se1, se2))/len(se1)


def fix_parameters_SequenceMatcher(se1, se2):
    seq = SequenceMatcher(None, se1, se2)
    return seq.ratio()


STRING_COMPARISON_METHODS = {
    'LEV': normalize_levenshtein_distance,
    'DAM-LEV': normalize_damerau_levenshtein_distance,
    'HAM': normalize_hamming_distance,
    'JAR': jaro_distance,
    'JAR-WIN': jaro_winkler,
    'PYTHON': fix_parameters_SequenceMatcher,
    'KENDALL': metrics.kendalltau_distance,
    'SPEARMAN': metrics.spearmanft,
    'G': metrics.metric_g,
    'M': metrics.metric_m,
    'T': metrics.metric_T
}


def find_distance(visual, urls, snippets, titles, search_engines,
                  query_category, method, evolution, N, weight_a,
                  weight_b, weight_c):
    """
    Calculate the "distance" of sorting of results of two search engines for
    each query and for each date. The results of two search engines converted
    to two strings, where each letter corresponds to a result of set of all
    results. The distance of strings is calculated with the given method.

    :param visual: A object of Visualization class.
    :param urls: A dictionary keyed by search engine, date and query containing
    the N urls of results.
    :param snippets: A dictionary keyed by search engine containing the N
    snippets of results.
    :param titles: A dictionary keyed by search engine containing the N titles
    of results.
    :param search_engines: List with exactly two search engines for comparing.
    :param query_category: The category of queries for comparing of
     search engines.
    :param method: The method for comparing of strings.
    :param evolution: If True, display the similarity of two search engines
     over time.
    :param N: Number of top results for each query.
    :param weight_a: The weight of transpositions.
    :param weight_b: The weight of snippets.
    :param weight_c: The weight of titles.
    """
    days = urls[search_engines[0]].keys()
    number_of_days = len(days)
    random_day = next(iter(days))
    number_of_queries = len(urls[search_engines[0]][random_day])

    distances = (
        np.zeros((number_of_days, number_of_queries, len(weight_a)))
        if isinstance(weight_a, np.ndarray)
        else np.zeros((number_of_days, number_of_queries, 1))
    )

    for d, date in enumerate(sorted(urls[search_engines[0]])):
        for q, query in enumerate(urls[search_engines[0]][date]):
            letters = string.ascii_lowercase[0:2 * N]
            unique_urls = (
                set(urls[search_engines[0]][date][query]) |
                set(urls[search_engines[1]][date][query])
            )

            url_to_char = {url: letters[i] for i, url in
                           enumerate(unique_urls)}
            se0_chars = [url_to_char[url] for url in
                         (urls[search_engines[0]][date][query])]
            se1_chars = [url_to_char[url] for url in
                         (urls[search_engines[1]][date][query])]

            if method == 'T':
                snippets0 = {(date, query): [snippets[search_engines[0]]
                             [(i + 1, date, query)].encode('utf-8')
                             for i in range(N)]}
                snippets1 = {(date, query): [snippets[search_engines[1]]
                             [(i + 1, date, query)].encode('utf-8')
                             for i in range(N)]}
                titles0 = {(date, query): [titles[search_engines[0]]
                           [(i + 1, date, query)].encode('utf-8')
                           for i in range(N)]}
                titles1 = {(date, query): [titles[search_engines[1]]
                           [(i + 1, date, query)].encode('utf-8')
                           for i in range(N)]}

                distances[d, q, :] = STRING_COMPARISON_METHODS[method](
                    ''.join(se0_chars),
                    ''.join(se1_chars),
                    snippets0, snippets1, titles0, titles1,
                    weight_a, weight_b, weight_c)
            else:
                distances[d, q] = STRING_COMPARISON_METHODS[method](
                    ''.join(se0_chars),
                    ''.join(se1_chars))
    if visual is None:
        return distances

    if evolution:
        visual.plot_day_similarity(distances, search_engines[0] + '-'
                                   + search_engines[1], weight_a,
                                   weight_b, weight_c)
    else:
        visual.plot_heatmap(distances[:, :, 0], ['Queries', 'Days'],
                            search_engines[0] + '-' + search_engines[1]
                            + '(' + query_category + ')')
    return distances

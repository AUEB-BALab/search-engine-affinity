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

from collections import OrderedDict, defaultdict
import os
from os.path import join, isfile, expanduser
import json
import glob
import io
import re
import requests
import jsonschema
from os import listdir
from jsonschema import validate


SCHEMA = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "title": "Query categories",
    "type": "object",
    "properties": {
        "categories": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "results": {
            "type": "string"
        },
    },
    "additionalProperties": {
        "type": "array",
        "items": {
            "type": "string"
        }
    },
    "required": ["categories", "results"]
}


PROJECT_DIR = ".."
CONFIG_FILE = '.config.sea'
DEFAULT_FILENAME = join(PROJECT_DIR, CONFIG_FILE)

N_RESULTS = 10


class NoResult(object):

    def encode(self, encode):
        """ Just a stub. """
        return self


NO_RESULT = NoResult()


class SEAnalysisException(Exception):
    """ Custom exception thrown by this package. """
    pass


def load_config(path=None):
    """
    This function load the config file where queries and their categories
    are defined.

    This config must be a valid JSON file as well as must be valid according
    to the JSON schema.

    SCHEMA = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "Query categories",
        "type": "object",
        "properties": {
            "categories": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "results": {
                "type": "string"
            },
        },
        "additionalProperties": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "required": ["categories", "results"]
    }
    """
    path = DEFAULT_FILENAME if not path else path
    if not isfile(path):
        msg = 'Given path {!r} is not a file'.format(path)
        raise SEAnalysisException(msg)

    with open(path) as data_file:
        data = json.load(data_file)
    try:
        validate(data, SCHEMA)
        return data
    except jsonschema.exceptions.ValidationError as e:
        raise SEAnalysisException(
            'Your configuration file is not valid, {!s}'.format(e.message))


def load_document(path):
    """
    Loads document of results in a JSON format.

    :param path: Path to document of results.

    :return: JSON representing results of a search engine.
    """
    with io.open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_url(document, query_category, se, index=0, edit=True, redirection=False):
    """
    Get a string of top URL of a document.

    :param document: Document of results.
    :param se: Search engine which returned the specified document.
    :param index: Index of result.
    :param edit: Edit the string of URL.
    :param redirection: Find possible redirections on retrieved URL.

    :return: String of URL.
    """
    try:
        if 'google' in se and query_category != 'News':
            url = document.get('items')[index].get('link')

        else:
            url = document[index].get('url')
    except IndexError:
        return NO_RESULT

    if redirection:
        r = requests.get(url)
        if r.history and (r.history[-1].status_code == requests.codes.moved or
                          r.history[-1].status_code == requests.codes.found or
                          r.history[-1].status_code == requests.codes.other):
            url = r.url

    if edit:
        url = re.sub('^https?://'
                     '|home$'
                     '|(external/)?index.(htm|asp|jsp)\?.*$', '', url)
    return url.lower()


def get_snippets(document, query_category, se, N=10):
    """
    Get a string of the concatenated snippets of a document.

    :param document: Document of results.
    :param se: Search engine which returned the specified document.
    :param N: Number of results to take their snippets.

    :return: String of concatenated snippets.
    """
    if N > N_RESULTS:
        raise SEAnalysisException('The number of top results must be up to 10')

    if 'google' in se:
        if query_category!='News':
            return ' '.join([item.get('snippet')
                             for item in document.get('items')[:N]])
        else:
            return ' '.join([item.get('abstract') for item in document[:N]])
    else:
        return ' '.join([item.get('description') for item in document[:N]])


def get_snippet(document, query_category, se, index=0):
    """
    Get the snippet of the n in order result taken from a specific search
    engine.

    :param document: Document of results.
    :param se: Search engine which returned the specified document.
    :param index: Index of result.

    :return: Snippet.
    """
    try:
        if 'google' in se:
            if query_category!='News':
                return document.get('items')[index].get('snippet')
            else:
                return document[index].get('abstract')
        else:
            return document[index].get('description')
    except IndexError:
        return NO_RESULT


def get_titles(document, query_category, se, N=10):
    """
    Get a string of the concatenated titles of a document.

    :param document: Document of results.
    :param se: Search engine which returned the specified document.
    :param N: Number of results to take their snippets.

    :return: String of concatenated titles.
    """
    if N > N_RESULTS:
        raise SEAnalysisException('The number of top results must be up to 10')

    if 'google' in se and query_category!='News':
            return ' '.join([item.get('title')
                             for item in document.get('items')[:N]])
    elif 'bing' in se and query_category=='News':
        return ' '.join([item.get('name') for item in document[:N]])
    else:
        return ' '.join([item.get('title') for item in document[:N]])


def get_title(document, query_category, se, index=0):
    """
    Get the title of the n in order result taken from a specific search
    engine.

    :param document: Document of results.
    :param se: Search engine which returned the specified document.
    :param index: Index of result.

    :return: Title.
    """
    try:
        if 'google' in se and query_category!='News':
            return document.get('items')[index].get('title')
        elif 'bing' in se and query_category == 'News':
            return document[index].get('name')
        else:
            return document[index].get('title')
    except IndexError:
        return NO_RESULT


def get_query_se(path, se=False):
    """
    Get query and search engine based on the path to the results
    document.

    :param path: Path to a results document.
    :param se: True if we want to take the search engine of
    query result; False otherwise.

    :return: Query and date. If `se` is True then the search
    engine is returned too.
    """
    start = 1 if path[0] == os.sep  else 0
    date, _, search_engine, fname = tuple(path.split(os.sep)[start:])
    return (fname.rsplit("-", 1)[0], date, search_engine) if se\
        else (fname.rsplit('-', 1)[0], date)


def get_result_dirs(results_dir, query_category, search_engines, date=False):
    """
    Gets a list of directories of the results based on the given query category
    and search engines.

    :param results_dir: Directory where results for every query and search
    engine are located.
    :param query_category: Category of queries.
    :param search_engines: List of search engines.
    :param date: Create dictionary dirs[date] of results directories.

    :returns: Dictionary or list of results directories.
    """
    if date:
        dirs = dict()
        for date in listdir(results_dir):
            dirs[date] = sum((glob.glob(
                    join(results_dir, date, query_category, se, '*'))
                              for se in search_engines), [])
    else:
        dirs = sum((glob.glob(join(results_dir, '*', query_category, se, '*'))
                    for se in search_engines), [])
        dirs.sort()
    return dirs


def set_snippets(snippets, document, query_category, se, date, query, N, per_day):
    """
    This function sets the snippets to the given dictionary according to the
    given dataset design.

    For example, each query results can be a different dataset instance or
    an instance can include the snippets for all results of a query.

    :param snippets: Dictionary keyed by search engines containing the
    snippets of results.
    :param document: JSON object of the retrieved results of a query.
    :param se: Search engine which returned the given result.
    :param date: Date when results was retrieved.
    :param query: Query associated with this result.
    :param N: Number of retrieved results per query.
    :param per_day: True if a dataset instance includes all results of a
    query.

    :returns: Initialized dictionary keyed by search engine containing the
    snippets of results.
    """
    if not per_day:
        for i in range(N):
            snippets[se][(i + 1, date, query)] = get_snippet(
                document, query_category, se, index=i)
        return snippets

    snippets[se][(date, query)] = (' ' + get_snippets(document, query_category, se, N=N))
    return snippets


def load_snippets(results_dir, query_category, search_engines, N=10,
                  per_day=True):
    """
    This method collects the snippets of the results of a specific date for
    every search engine.

    :param results_dir: Directory where results for every query and search
    engine are located.
    :param query_category: Query category.
    :param search_engines: List of search engines.
    :param N: Number of retrieved results per query.
    :param per_day: True if a dataset instance includes all results of a
    query.

    :returns: Initialized dictionary keyed by search engine containing the
    snippets of results.
    """
    dirs = get_result_dirs(results_dir, query_category, search_engines)

    snippets = {se: OrderedDict() for se in search_engines}
    for path in dirs:
        stripped_path = path.replace(results_dir, '', 1)
        document = load_document(join(results_dir, path))
        query, date, se = get_query_se(stripped_path, se=True)
        snippets = set_snippets(snippets, document, query_category, se, date, query, N,
                                per_day)
    return snippets


def set_titles(titles, document, query_category, se, date, query, N, per_day):
    """
    This function sets the titles to the given dictionary according to the
    given dataset design.

    For example, each query results can be a different dataset instance or
    an instance can include the titles for all results of a query.

    :param titles: Dictionary keyed by search engines containing the
    titles of results.
    :param document: JSON object of the retrieved results of a query.
    :param se: Search engine which returned the given result.
    :param date: Date when results was retrieved.
    :param query: Query associated with this result.
    :param N: Number of retrieved results per query.
    :param per_day: True if a dataset instance includes all results of a
    query.

    :returns: Initialized dictionary keyed by search engine containing the
    titles of results.
    """
    if not per_day:
        for i in range(N):
            titles[se][(i + 1, date, query)] = get_title(
                document, query_category, se, index=i)
        return titles

    titles[se][(date, query)] = (' ' + get_titles(document, query_category, se, N=N))
    return titles


def load_titles(results_dir, query_category, search_engines, N=10,
                  per_day=True):
    """
    This method collects the titles of the results of a specific date for
    every search engine.

    :param results_dir: Directory where results for every query and search
    engine are located.
    :param query_category: Query category.
    :param search_engines: List of search engines.
    :param N: Number of retrieved results per query.
    :param per_day: True if a dataset instance includes all results of a
    query.

    :returns: Initialized dictionary keyed by search engine containing the
    titles of results.
    """
    dirs = get_result_dirs(results_dir, query_category, search_engines)
    titles = {se: OrderedDict() for se in search_engines}
    for path in dirs:
        stripped_path = path.replace(results_dir, '', 1)
        document = load_document(join(results_dir, path))
        query, date, se = get_query_se(stripped_path, se=True)
        titles = set_titles(titles, document, query_category, se, date, query, N,
                                per_day)
    return titles


def load_urls(results_dir, query_category, search_engines, N=10):
    """
    This method collects the urls of the results of a specific date for
    every search engine.

    :param results_dir: Directory where results for every query and search
    engine are located.
    :param query_category: Query category.
    :param search_engines: List of search engines.
    :param N: Number of retrieved results per query.
    :param per_day: True if a dataset instance includes all results of a
    query.

    :returns: A dictionary keyed by search engine, date and query containing
    the N urls of results.
    """
    dirs = get_result_dirs(results_dir, query_category, search_engines)
    urls = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
    for path in dirs:
        stripped_path = path.replace(results_dir, '', 1)
        document = load_document(join(results_dir, path))
        query, date, se = get_query_se(stripped_path, se=True)
        for i in range(N):
            url = get_url(document, query_category, se, index=i)
            if url is not NO_RESULT:
                url = url.encode('ascii', 'ignore')
            urls[se][date][query].append(url)

    return urls

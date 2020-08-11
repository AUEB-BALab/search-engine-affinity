#! /usr/bin/env python
# coding=utf-8
import json
import io
import logging
import os
import random
import sys
import urllib
from os.path import join, expanduser
from datetime import datetime

import dryscrape
from bs4 import BeautifulSoup
import click
from googleapiclient.discovery import build
from py_ms_cognitive import PyMsCognitiveWebSearch, PyMsCognitiveNewsSearch

from seanalysis import utils
from seanalysis.cli.cmd import validate_query_categories
from seanalysis.utils import SEAnalysisException


reload(sys)
sys.setdefaultencoding('utf8')

logger = logging.Logger(__name__)

SUPPORTED_SEARCH_ENGINES = ['google', 'bing', 'duckduckgo']

DUCKDUCKGO_URL = 'https://duckduckgo.com'
TITLE_CLASS = 'result__a'
SNIPPET_CLASS = 'result__snippet js-result-snippet'
SNIPPET_NEWS_CLASS = 'result__snippet'
URL_CLASS = 'result__url js-result-extras-url'
URL_NEWS_CLASS = 'result__url'
RESULTS_DIV_CLASS = 'results js-results'
LIST_OF_RESULTS_CLASS = (
    'result results_links_deep highlight_d result--url-above-snippet'
)
RESULTS_DIV_NEWS_CLASS = 'results js-vertical-results'
LIST_OF_RESULTS_NEWS_CLASS = [
    'result result--news result--url-above-snippet',
    'result result--news result--img result--url-above-snippet'
]
TIMESTAMP_CLASS = 'None'
TIMESTAMP_NEWS_CLASS = 'result__timestamp'


REGULAR_CLASSES = (
    RESULTS_DIV_CLASS,
    LIST_OF_RESULTS_CLASS,
    SNIPPET_CLASS,
    URL_CLASS,
    TIMESTAMP_CLASS,
)
NEWS_CLASSES = (
    RESULTS_DIV_NEWS_CLASS,
    LIST_OF_RESULTS_NEWS_CLASS,
    SNIPPET_NEWS_CLASS,
    URL_NEWS_CLASS,
    TIMESTAMP_NEWS_CLASS,
)
dryscrape.start_xvfb()
session = dryscrape.Session()


def _get_content(query, query_category):
    """
    Get HTML content of a specific page of duckduckgo's website.

    :param query: Query used to get HTML page of results.

    :return: HTML content.
    """
    if query_category == "News":
        params = {'q': query, 'iar':'news', 'ia': 'news'}
    else:
        params = {'q': query, 'ia': 'about'}
    session.set_header('Cookie', 'ah=us-en%2Cus-en; l=us-en')
    session.visit(DUCKDUCKGO_URL + '/?' + urllib.urlencode(params))
    response = session.body()
    return response


def _get_title(div):
    """
    Get title of a page:

    :param div: HTML div element which contains query result.

    :return: Title of page.
    """
    return div.find('a', {'class': TITLE_CLASS}).get_text()


def _get_snippet(div, snippet_class=SNIPPET_CLASS):
    """
    Get snippet of page.

    :param div: HTML div element which contains query result.

    :return: Snippet of page.
    """
    return div.find('div', {'class', snippet_class}).get_text()


def _get_url(div, url_class=URL_CLASS):
    """
    Full url of page.

    :param div: HTML div element which contains query result.

    :return: Full url of page.
    """
    return div.find('a', {'class', url_class}).get('href')


def _get_timestamp(div,timestamp_class=TIMESTAMP_CLASS):
    """
    Timestamp for result

    :param div: HTML div element which contains query result.

    :return: timestamp of News article when query category is News
             else return None
    """
    if timestamp_class!='None':
        return div.find('span', {'class', timestamp_class}).get_text()
    else:
        return timestamp_class


def html_to_list(query, query_category):
    """
    Get a HTML page of duckduckgo's website with the results of a specific
    query.

    Filter content and get the url, snippet and title of top 10 results. Then,
    save them in a list of dictionaries.

    :param query: Query.

    :return: List of dictionaries. Each dictionary contains information of a
    specific result (title, url and snippet).
    """
    html_content = _get_content(query, query_category)
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []
    div_class, results_class, snippet_class, url_class, timestamp_class = (
        REGULAR_CLASSES
        if query_category != 'News'
        else
        NEWS_CLASSES
    )
    div = soup.findAll('div', {'class': div_class})[0]
    for result_div in soup.findAll('div', {'class': results_class}):
        result = {
            'title': _get_title(result_div),
            'description': _get_snippet(result_div,snippet_class=snippet_class),
            'url': _get_url(result_div,url_class=url_class),
            'timestamp': _get_timestamp(result_div,timestamp_class=timestamp_class)
        }
        results.append(result)
    return results


def validate_search_keys(conf):
    google_api_keys = conf.get('google-api-keys', None)
    bing_api_keys = conf.get('bing-api-keys', None)
    cse_key = conf.get('cse_key', None)

    if not google_api_keys:
        raise click.UsageError('google api keys not found')

    if not bing_api_keys:
        raise click.UsageError('bing api keys not found')

    if not cse_key:
        raise click.UsageError('cse key not found')

    return google_api_keys, bing_api_keys, cse_key


class SearchEngine(object):
    def get_results(self, queries, directory, query_category):
        """ This function gets a list of queries and retrieves the results for every query.

            The results of every query are stored inside the `directory` passes as an argument to this function.
            The results of every query are stored in a separate file.
        """
        raise NotImplementedError('get_results() must be implemented')


class Google(SearchEngine):
    """
    Class responsible for collecting results via google api.
    """

    SEARCH_COUNTER = 0
    QUOTA_LIMIT = 95
    API_INDEX = 0

    def __init__(self, api_keys, cse_key):
        self.api_keys = api_keys
        self.cse_key = cse_key

    def get_results(self, queries, directory, query_category):
        """
        Get top 10 results of all queries using the Google custom search API.

        The results returned by the API are in JSON format; this function
        stringifies them and writes them a corresponding file.

        :param queries: List of queries.
        :param directory: Directory to save results.
        """
        if query_category!="News":
            for query in queries:
                if self.SEARCH_COUNTER == self.QUOTA_LIMIT:
                    self.SEARCH_COUNTER = 0
                    self.API_INDEX += 1
                api_key = self.api_keys[self.API_INDEX]
                service = build("customsearch", "v1", developerKey=api_key)
                results = service.cse().list(q=query, cx=(self.cse_key)[0], num=10, gl="us").execute()
                json_str = json.dumps(results, encoding="utf-8", ensure_ascii=False, indent=2)
                write_results(query, json_str, 'google', directory)
                self.SEARCH_COUNTER += 1
        else:
            pass

class Bing(SearchEngine):

    def __init__(self, api_keys):
        self.api_keys = api_keys

    def get_results(self, queries, directory, query_category):
        """
         Class responsible for collecting results via bing api.

        Get top 10 results of all queries using the Bing search API.

        The results returned by the API is a list of Objects; this
        function converts them in stringified JSON format and writes them in
        a corresponding file.

        :param queries: List of queries.
        :param directory: Directory to save results.
        """

        # for PyMsCognitiveWebSearch to be working please adjust accordingly
        # https://github.com/tristantao/py-ms-cognitive/issues/33

        for query in queries:
            if query_category!="News":
                bing_search_service = PyMsCognitiveWebSearch(random.choice(self.api_keys), query,
                                                         custom_params={"mkt": "en-US"})
            else:
                bing_search_service = PyMsCognitiveNewsSearch(random.choice(self.api_keys), query,
                                                         custom_params={"mkt": "en-US"})
            result_list = bing_search_service.search(limit=50, format='json')  # 1-10
            json_str = json.dumps([result for result in result_list],
                                  default=lambda o: o.__dict__, indent=2,
                                  encoding='utf-8', ensure_ascii=False)
            result_len = len(result_list)
            if result_len < 10:
                msg = 'Bing returned {!d} results for the query {!r}'
                logger.error(msg.format(result_len, query))
            write_results(query, json_str, 'bing', directory)


class Duckduckgo(SearchEngine):
    def get_results(self, queries, directory, query_category):
        """
        Get top results of all queries using the duckduckgo's parser.

        The results returned by the parser is a list of dictionaries; this
        function converts them in stringified JSON format and writes them in
        a corresponding file.

        :param queries: List of queries.
        :param directory: Directory to save results.
        """
        for query in queries:
            result_list = html_to_list(query, query_category)
            json_str = json.dumps([result for result in result_list],
                                  indent=2, encoding='utf-8', ensure_ascii=False)
            result_len = len(result_list)
            if result_len < 10:
                msg = 'Duckduckgo returned {!s} results for the query {!r}'
                logger.error(msg.format(result_len, query))
            write_results(query, json_str, 'duckduckgo', directory)


def write_results(query, results, se, directory):
    """
    Write results in a JSON file.

    :param query: Query which was used to get results.
    :param results: Results returned by the API in stringified JSON format.
    :param se: Search engine.
    :param directory: Directory to save results.
    """
    folder = join(directory, se)
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_name = query + '-' + se + '_results.json'
    with io.open(join(directory, se, file_name),
                 'w', encoding='utf-8') as outfile:
        outfile.write(results.decode('utf-8'))


def mkdir_results(root_directory, query_category):
    """
    Make a directory where the results of a specific day are included.

    :param query_category: Categories of queries, e.g. trends.
    :return: Directory.
    """
    date = datetime.now().strftime('%m-%d-%Y')
    directory = join(root_directory, date + '-results', query_category)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def validate_search_engines(search_engines):
    """
    Validate if given search engines are supported by this package.

    :param search_engines: List of given search_engines.
    :raises: SEAnalysisException if at least one of the given search engines
    is not supported by this package.
    """
    invalid_search_engines = set(search_engines).difference(
        SUPPORTED_SEARCH_ENGINES)
    if invalid_search_engines:
        raise SEAnalysisException('Unsupported search engine %s' % (
            invalid_search_engines.pop()))


def get_search_engine_results(query_categories, search_engines,
                              root_directory, google_api_keys, bing_api_keys, cse_key):
    """
    Collect the results of queries for the specified search_engines.

    This function is responsible for retrieving the results of queries which
    are included to the list of the given categories for all the specified
    search engines.

    Moreover, the results are written to seperate files included in the given
    root directory.

    "param query_categories: List of query categories.
    :param search_engines: List of search engines.
    :param root_directory: Directory to save the result files.
    """

    # validate_search_engines(search_engines)
    # google_search = GoogleSearch(kwargs['google_api_keys'], kwargs['cse_key'])
    # bing_search = BingSearch(kwargs['bing_api_keys'])
    g = Google(google_api_keys, cse_key)  # It might require the api keys for the google api.
    b = Bing(bing_api_keys)
    d = Duckduckgo()

    SEARCH_ENGINES_RESULTS = {
        'google': g.get_results,
        'bing': b.get_results,
        'duckduckgo': d.get_results,
    }

    for query_category, queries in query_categories.iteritems():
        # Create directory for each query category
        directory = mkdir_results(root_directory, query_category)
        print ('Processing the {!r} category'.format(query_category))
        for search_engine in search_engines:

            print ('Retrieving the results from {!s}'.format(search_engine))

            try:
                SEARCH_ENGINES_RESULTS[search_engine](queries, directory, query_category)
                logger.info('All results for the queries of category %s were '
                            'retrieved successfully by %s.' % (
                                repr(query_category), repr(search_engine)))
            except Exception as e:
                logger.error(e)
                raise SEAnalysisException(e)


def init_log_handlers(logfile):
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    error_handler = logging.FileHandler(logfile)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)


DEFAULT_LOG_FILE = 'results.log'


@click.command()
@click.option('--search_engines', '-s', help='search engines for the analysis',
              type=str, required=True)
@click.option('--categories', help='categories of the queries',
              type=str)
@click.option('--config', '-c', help='path to config file',
              type=str, required=False)
@click.option('--logs', '-l', help='path to log file',
              type=str, required=False)

def search(search_engines, categories, config, logs):
    try:
        # Load config file .config.sea
        conf = utils.load_config(config)
    except utils.SEAnalysisException as e:
        raise click.UsageError(click.style(str(e), fg='red'))

    results_dir = conf['results']
    query_categories = conf['categories'] #dictionary
    init_log_handlers(logs or DEFAULT_LOG_FILE)

    search_engines = search_engines.split(',')
    categories = categories.split(',')

    # Create the dictionary of {category name : category queries}
    if categories != ['all']:
        query_categories = {cat_name: cat_queries
                            for cat_name, cat_queries in conf['categories'].iteritems()
                            if cat_name in categories}

    google_api_keys, bing_api_keys, cse_key = validate_search_keys(conf)

    try:
        get_search_engine_results(
            query_categories, search_engines, results_dir,
            google_api_keys=google_api_keys,
            bing_api_keys=bing_api_keys, cse_key=cse_key)
    except utils.SEAnalysisException as e:
        raise click.UsageError(click.style(str(e), fg='red'))


if __name__ == '__main__':
    search()

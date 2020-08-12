#! /usr/bin/env python3

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

import argparse
import glob
import json
import io
import os
import pickle

import requests
from bs4 import BeautifulSoup



def load_document(filename):
    with io.open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_document(filename, document):
    json_str = json.dumps(document, ensure_ascii=False, indent=2)
    with io.open(filename, 'w', encoding='utf-8') as outfile:
        outfile.write(json_str)


def load_cache(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except IOError:
        return {}


def save_cache(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def resolve_regular_redirection(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=10)
    except:
        return url
    if r.history and (
            r.history[-1].status_code == requests.codes.moved or
            r.history[-1].status_code == requests.codes.found or
            r.history[-1].status_code == requests.codes.other):
        url = r.url
    return url


def resolve_wikipedia_redirection(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    element = soup.find('link', {'rel': 'canonical'})
    if element is None:
        return url
    return element.get('href')


def resolve_redirection(url, is_wikipedia=False):
    if not is_wikipedia:
        return resolve_regular_redirection(url)
    return resolve_wikipedia_redirection(url)


def get_urls(document, N, is_google=False):
    if is_google:
        return [x.get('link') for x in document['items']][:N]
    else:
        return [x['url'] for x in document][:N]


def set_new_urls(document, new_urls, is_google):
    for i, url in enumerate(new_urls):
        if url is None:
            raise Exception('URL is None')
        if is_google:
            document['items'][i]['link'] = url
        else:
            document[i]['url'] = url


def create_parent_directories(filename):
    segments = filename.split(os.sep)
    dirs = os.sep.join(segments[:-1])
    os.makedirs(dirs, exist_ok=True)


def canonicalize_urls(results_dir, output_dir, N=5, ignore_days=[],
                      ignore_categories=[],
                      in_cache_file=None,
                      out_cache_file=None):
    cache = load_cache(in_cache_file) if in_cache_file else {}
    ignore_days = [x + '-results' for x in ignore_days]
    try:
        for filename in glob.glob(results_dir + '/*/*/*/*'):
            segments = filename.split(os.sep)
            day_result = segments[2]
            category = segments[3]
            if day_result in ignore_days or category in ignore_categories:
                continue
            document = load_document(filename)
            is_google = 'google' in filename and 'News' not in filename
            urls = get_urls(document, N, is_google)
            new_urls = []
            for url in urls:
                is_wikipedia = 'en.wikipedia.org/wiki/' in url
                new_url = cache.get(url)
                if new_url is None:
                    new_url = resolve_redirection(url, is_wikipedia)
                    cache[url] = new_url
                if url != new_url:
                    print ('{!s} -> {!s}'.format(url, new_url))
                new_urls.append(new_url)
            set_new_urls(document, new_urls, is_google)
            new_filename = filename.replace(results_dir, output_dir)
            create_parent_directories(new_filename)
            save_document(new_filename, document)
    finally:
        # In a case of an unexpected exception
        # save cache to the specified file.
        if out_cache_file:
            save_cache(cache, out_cache_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results',
                        type=str,
                        help='Directory of results')
    parser.add_argument('-n', type=int, default=5,
                        help='Number of results to inspect')
    parser.add_argument('--ignore-days',
                        type=str,
                        help='List of days to ignore')
    parser.add_argument('--ignore-categories',
                        type=str,
                        help='List of query categories to ignore')
    parser.add_argument('--out-cache',
                        type=str,
                        help='Path to save URL cache')
    parser.add_argument('--in-cache',
                        type=str,
                        help='Load cache from the given directory')
    parser.add_argument('--output', '-o',
                        type=str,
                        required=True,
                        help='Directory to store the new documents')
    args = parser.parse_args()
    ignore_days = args.ignore_days.split(',') if args.ignore_days else []
    ignore_categories = (
        args.ignore_categories.split(',')
        if args.ignore_categories else []
    )

    canonicalize_urls(args.results, args.output, ignore_days=ignore_days,
                      ignore_categories=ignore_categories,
                      N=args.n, in_cache_file=args.in_cache,
                      out_cache_file=args.out_cache)


if __name__ == '__main__':
    main()

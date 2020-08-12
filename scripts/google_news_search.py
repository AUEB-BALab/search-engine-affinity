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

import os
import subprocess
import json
from datetime import datetime
from os.path import join
import time
import random

def mkdir_results(results_dir):
    """
    Make a directory where the results of a specific day are included.

    :param query_category: Categories of queries, e.g. trends.
    :return: Directory.
    """
    date = datetime.now().strftime('%m-%d-%Y')
    directory = join(results_dir, date + '-results', 'News/google')
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

conf_file_path = '.config.sea'
with open(conf_file_path) as data_file:
    conf = json.load(data_file)
    news_queries = (conf['categories']['News'])
    results_dir = conf['results']

# Create directory for each query category
directory = mkdir_results(results_dir)

for query in news_queries:
    command = 'googler -c us -l en -x -n 10 --json -N ' + query + ' > '+directory+'/'+query.replace(" ", "\ ")+'-google_results.json'
#    command = 'pip freeze > '+directory+'/'+query.replace(" ", "\ ")+'-google_results.json'
    subprocess.call(command, shell=True)  # shell=True hides console window
    time.sleep(random.randint(1,3))

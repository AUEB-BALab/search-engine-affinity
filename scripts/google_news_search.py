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

conf_file_path = '/home/user/.config.sea'
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

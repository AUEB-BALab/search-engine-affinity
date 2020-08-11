Search Engine Similarity Analysis: A Combined Content and Rankings Approach
===========================================================================


Are search engines different? The search engine wars are a
favourite topic of on-line analysts, as two of the biggest companies
in the world, Google and Microsoft, battle for dominance of the web
search space. Differences in search engine popularity can be
explained because one of them gives better results to the users, or
by other factors, such as familiarity with the most popular first
engine, peer imitation, or force of habit. In this work we present a
thorough analysis of the affinity of the two major search engines,
while also adding a third one, DuckDuckGo, which goes to great
lengths to emphasize its privacy-friendly credentials. To do the
analysis, we collected response data using a comprehensive set of
queries for a period of one month. Then we applied a number of
different similarity methods, while also using a new measure, which
we developed specifically for the task that leverages both the
content and the ranking of search responses. The results show that
Google stands apart, but Bing and DuckDuckGo are largely
indistinguishable from each other. It therefore appears that, while
the difference between Google and the rest can be explained by the
different results they produce, search results cannot account for
the difference in market share between Bing and DuckDuckGo.
Moreover, our work, via the new measure we propose, contributes also
to the top `k` lists comparison literature and the application of
machine learning techniques to search engine comparisons.


# The seanlz Tool

To estimate the degree of similarity between the web results of
search engines we implemented a command-line tool which computes
the similarity with various ways. This tools takes as input the web
results of search engines produced by multiple queries for multiple days.

## Installation

In a [virtualenv](http://bit.ly/2nVVhA3), install all required packages
via pip:

```console
pip install -r requirements.txt
```

and, then install command tool:

```console
python setup.py install
```

### Dataset

To get dataset used in our work, run:
```
wget https://pithos.okeanos.grnet.gr/public/Um13GYS9koXLg5Y6MrvBz2
```

## Use

The `seanlz` command provides you multiple methods to compute similarity
of search engines with the following sub-commands:
- `cont`: Evaluates the results of search engines by exploiting the
   snippets of results and using machine learning, and tensor
   decomposition techniques.
- `rank`: Measures the similarity of a pair of search engines, using
  ranking-based methods.
- `metrict`: Computes the similarity between two search engines,
  using the metric T.

### Config

The `seanlz` tool needs a configuration file which defines which queries
are used in your dataset. These queries are grouped into categories, thus,
you can make an analysis based on the category.

This configuration file must follow a `JSON`. A valid schema for this
configuration file is the following:

```json
{
    "categories": {
        "Query Category 1": ["Query1", "Query2"],
        "Query Category 2": ["Query3", "Query4"],
    },
    "results": "directory/to/your/dataset"
}
```

The configuration file used for our work is `.config.sea` located on the
root of this repository.

# CLI Reference

## seanalz

### Description
Estimates the degree of similarity between the results of multiple
search engines produced from queries grouped into categories.

### Synopsis

```
seanlz [options] <sub-command>
```

### Options

- `--config`: (optional) The location of configuration file. This location
  can also be specified by an enviroment variable named `SESIM_CONFIG`.
  Default location is `~/.config.sea`.
- `-s`, `--search_engines`: (required) The search engines for the analysis
  (seperated by ',').
- `--categories` (optional): The categories of the queries for the analysis
  (seperated by ','). By default, sub-command inspects all categories
  specified in the configuration file.
- `--merge`: (flag option) Display analysis of query categories in a single
             diagram.
- `-N`: (int) Number of retrieved results per query. Default 10.


## cont

### Description
Evaluates the results of search engines by exploiting the snippets of
results and using machine learning, and tensor decomposition techniques.

### Synopsis

```console
seanlz [options] cont [options]
```

### Options

- `method`: (`cmp` or `clf` -- required) Strategy used to evaluate the results.
  The `cmp` denotes that results are split into components to examine the
  similarity of search engines. The `clf` indicates that a classifier is
  used a mean of similarity between search engines.
- `-m`, `--model`: (`tensor` | `lda` |`query` | `se` -- required)
  Actual model used to estimate the similarity.
  - `tensor`: A four-mode tensor (query, search engine, day, result) is
  constructed, and CP Alternating Poisson Regression algorithm is
  applied on it.
  - `lda`: Latent Dirictlet Allocation algorithm is used.
  - `query`: Classification model for predicting queries is used.
  - `se`: Classification model for predicting search engines is used.
- `-c` (required) Model-specific configuration. Key-value
  pairs, seperated by `=`.

  `tensor` and `lda` models need the number of components to be specified.

  Example:

  ```console
  seanlz -s foo,bar cont --method cmp -m lda -c components=10
  ```

  `query` model need the following configuration options:
  - `evaluation`: Evaluation method for classification model. Currently, only
  `CrossLearnCompare` method is supported.
  - `classfier`: Classifier for predicting queries (e.g. SVC).

  Example:

  ```console
  seanlz -s foo,bar cont --method clf -m query -c evalution=clc -c classifier=SVC
  ```

  `se` model needs:
  - `folds`: Number of folds for the cross-validation evaluation method.
  - `metric`: (roc | cm) Metric for computing the effectiveness of classification
  model (Receiver Operating Characteristic or Confusion Matrix).
  - `classifier`: Classifier for predicting search engines (e.g. SVC).
- `--vocabulary`: (int) Number of words of vocabulary to be used for the
                analysis. Default 100.
- `--per-day` / `--per-result`: Specify what the bag of words representation
  of results contains. The `--per-day` option denotes that all top results are
  taken into account, whereas the `--per-result` indicates that every single
  result is converted into a bag of words representation.

## rank

### Description
Measures the similarity of a pair of search engines, using
ranking-based methods

### Synopsis
```
seanlz [options] rank [options]
```
### Options

- `--metric`: (required) Ranking-based method used to measure similarity.
  Methods:
  - `LEV`: Levenshtein distance.
  - `DAM-LEV`: Damerau-Levenshtein distance.
  - `HAM`: Hamming distance
  - `JAR`: Jaro distance.
  - `JAR-WIN`: Jaro-Winkler distance.
  - `KENDALL`: Kendal-Tau distance.
  - `SPEARMAN`: Spearman's Footrule distance.
  - `G`: G distance.
  - `M`: M distance.
  - `PYTHON`: Uses python's [difflib.SequenceMatcher](https://docs.python.org/2/library/difflib.html#difflib.SequenceMatcher)

## metrict

### Description
Computes the similarity between two search engines, using the metric T.

### Synopsis

```
seanlz [options] metrict [options]
```

### Options

- `--evol`: (flag option) Display the similarity of two search engines over time. 
- `-a`: (float, between 0-1) The weight attached to the penalty due to
  snippets. It accepts more than one values (seperated by ','), **only** with
  the `--evol` option. Default is 0.8.
- `-b`: (float, between 0-1) The weight attached to the penalty due to
  titles. It accepts more than one values (seperated by ','), **only** with
  the `--evol` option. Default is 1.0.
- `-c`: (float, between 0-1) The weight attached to the penalty due to
  transpositions. It accepts more than one values (seperated by ','),
  **only** with the `--evol` option. Default is 0.33.

# coding=utf-8
from collections import namedtuple
import string
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# A special character used to identify the query associated with
# a string of concatenated snippets, e.g. Athens~foo bar blah blah...
SPLIT_CHAR = '~'

# Load stop words.
punctuation = list(string.punctuation)
punctuation.append('...')

# Add punctuation to stop words.
stops = list(ENGLISH_STOP_WORDS) + punctuation


BagOfWordsOutput = namedtuple('BagOfWordsOutput', ['matrix', 'se'])


def get_top_words(bow, words, N=100):
    """
    Calculate the `N` most frequent words of a search engine
    bag of words representation matrix.

    :param bow: Bag of words matrix of search engine.
    :param words: List with feature names of Bag of words.
    :param N: Number of top words.
    :return: Top `N` words list.
    """
    sum_of_columns = np.array(bow.sum(axis=0))
    indexes = np.argsort(sum_of_columns)[::-1][:N]

    return [words[index] for index in indexes]


def transform(bow, words, union):
    """
    Construct the new bag of words with union of these two
    top 100 words list as feature space.

    :param bow: Bag of words of search engine.
    :param words: List with feature names of Bag of words.
    :param union: Union of these two top words list.

    :return: Bag of words binary array with new feature space.
    """
    train = np.zeros((bow.shape[0], len(union)))
    for i, word in enumerate(union):
        try:
            index = words.index(word)
            train[:, i] = bow[:, index]
        except ValueError:
            pass

    return (train > 0).astype(int)


def add_query_term(se_snippet):
    """
    Add the query term and the SPLIT_CHAR in the beginning of
    snippets.

    :param se_snippet: A dictionary with the tuple (date, query) as
    key and snippets as value.

    :return: A dictionary with the tuple (date, query) as key and as
    value the snippets with the query term and the SPLIT_CHAR in the
    beginning.
    """
    for key, snippet in se_snippet.items():
        if len(key) == 2:
            date, query = key
            se_snippet[(date, query)] = SPLIT_CHAR.join((query, snippet))
        else:
            i, date, query = key
            se_snippet[(i, date, query)] = SPLIT_CHAR.join((query, snippet))
    return se_snippet


class BagOfWords:

    def __init__(self, se_snippets, remove_query_term=True):
        self.se_snippets = se_snippets
        self.remove_query_term = remove_query_term
        self.se_bows = self._vectorize_snippets()

    def get_queries(self):
        """
        Gets the list of queries associated with the retrieved documents
        and theirs bag of words representation.

        :return: List of queries.
        """
        random_se = next(iter(self.se_snippets.keys()))
        return [ x[-1] for x in self.se_snippets[random_se].keys() ]

    def get_indexes(self):
        """
        Gets the list of the indexes of the results which are associated with
        the retrieved documents and theirs bag of words representation.

        :return: List of indexes.
        """
        random_se = next(iter(self.se_snippets.keys()))
        return [ x[0] for x in self.se_snippets[random_se].keys() ]

    def _vectorize_snippets(self):
        """
        This function vectorizes a list of snippets data associated
        with a specific search engine for every date and queries.

        :return: List of Bag of words representation objects.
        One for each search engine.
        """
        bow_se = []
        for se, se_snippet in self.se_snippets.items():
            vectorizer = CountVectorizer(stop_words='english',
                                         analyzer='word',
                                         binary=False,
                                         tokenizer=self.tokenize)
            edit_se_snippet = add_query_term(se_snippet)
            X = vectorizer.fit_transform(edit_se_snippet.values())
            bow_se.append((X.toarray(), se, vectorizer))
        return bow_se

    def tokenize(self, query_sentence):
        """
        Get tokens of a sentence, after eliminating all stop words.

        :param query_sentence: String of concatenated snippets.
        It starts with the query related to it.

        :return: List of words.
        """
        split_snippet = query_sentence.split(SPLIT_CHAR, 1)
        query = split_snippet[0]

        assert query != query_sentence

        sentence = query_sentence.replace(query + SPLIT_CHAR, '', 1)
        query_tokens = word_tokenize(query.lower())

        if self.remove_query_term:
            words = [word for word in word_tokenize(sentence.lower())
                     if word not in stops and not word.isdigit()
                     and word not in query_tokens
                     and query.lower() not in word]
        else:
            words = [word for word in word_tokenize(sentence.lower())
                     if word not in stops and not word.isdigit()]

        return words if words else [u" "]

    def build_bows(self, N=100):
        """
        This method represents the results of search engine associated with
        a category of queries, e.g. manuals.

        Method constructs binary bag of words representation using the union of
        `N` top words of each search engine. This means that all terms are
        taken account of

        :param N: Number of top words.

        :return List of bag of words representation arrays.
        """
        union = []

        for bow, se, vectorizer in self.se_bows:
            if N is not None:
                union += get_top_words(
                    bow, vectorizer.get_feature_names(), N)
            else:
                union += vectorizer.get_feature_names()

        union = list(set(union))
        union.sort()
        return [BagOfWordsOutput(
            transform(bow, vectorizer.get_feature_names(), union), se)
            for bow, se, vectorizer in self.se_bows]

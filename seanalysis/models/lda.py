import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from seanalysis.models import Model


class LDA(Model):
    """
    This class represents a technique that uses Latent Dirictlet Allocation
    algorithm to allocate results of search engines into topics.

    The goal of this technique is to find out the similarity of search engines.
    Obviously, if one topic contains results produced by a uniformly number
    of search engines, then we can infer that search engines are similar
    as they equally contribute to the topic.

    On the other hand, when results produced by different search engines are
    allocated to different topics, then we can infer that search engines are
    not similar.
    """
    def __init__(self, bows, queries, components=20):
        self.X = None
        self.se_dist = None
        self.query_dist = None
        self.bows = bows
        self.components = components
        self._search_engines = [bow.se for bow in bows]
        self._se_query = None
        self._documents = None
        self._queries = queries

    def construct(self):
        """
        Construct dataset to be used by LDA.

        Dataset contains the bag of words representation of a result produced
        by a specific search engine on a specific date.
        """
        self._documents = np.vstack([bow.matrix for bow in self.bows])
        se_query = []
        for bow in self.bows:
            se = [bow.se] * bow.matrix.shape[0]
            se_query += zip(se, self._queries)
        self._se_query = np.asarray(se_query)

    def evaluate(self):
        """
        Use LDA to allocate results of search engines into topics.

        Then find, the query and search engine frequency distribution on each
        topic.

        :return: 1) Numpy array (n_topics, search_engines) with frequency
                 distribution of each search engine on each topic.
                 2) List of list of tuples (query, dist) with the frequency
                 distribution of each query on each topic.
        """
        lda = LatentDirichletAllocation(n_components=self.components, max_iter=20,
                                        learning_method='online',
                                        learning_offset=10., random_state=1)
        self.X = lda.fit_transform(self._documents)
        doc_allocation = np.argmax(self.X.transpose(), axis=0)
        docs = [np.where(doc_allocation == topic_id)[0].tolist()
                for topic_id in range(self.components)]
        self.se_dist, self.query_dist = self._topic_dist(docs)

    def plot(self, visual, query_category):
        """
        This method plots the distribution of every search engine to every
        topic.

        If the search engines are similar, it is expected that their
        distribution to every cluster is also similar.

        :param visual: Visualization obj responsible for the plotting of
        resutlts.
        :param query_category: Category of queries which were used for the
        analysis.
        """
        visual.draw_se_clusters(self.se_dist, query_category,
                                self._search_engines)

    def _topic_dist(self, docs):
        """
        Find the query and search engine frequency distribution on each topic.

        :return: 1) Numpy array (n_topics, search_engines) with frequency
                 distribution of each search engine on each topic.
                 2) List of list of tuples (query, dist) with the frequency
                 distribution of each query on each topic.
        """
        non_empty_topics = [doc for doc in docs if doc]
        se_memberships = np.zeros(
            (len(non_empty_topics), len(self._search_engines)), dtype=float)
        query_dist = []
        for i, doc in enumerate(non_empty_topics):
            doc_len = len(doc)
            se, se_counts = np.unique(self._se_query[doc, 0],
                                      return_counts=True)
            queries, q_counts = np.unique(self._se_query[doc, 1],
                                          return_counts=True)
            query_dist.append(
                zip(queries, [float(x) / doc_len for x in q_counts]))
            se_indexes = [self._search_engines.index(s) for s in se]
            scores = [float(s) / doc_len for s in se_counts]
            se_memberships[i, se_indexes] = scores
        return se_memberships.transpose(), query_dist

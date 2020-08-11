import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity as cosine
from seanalysis import utils
from seanalysis.algorithms import bag_of_words as bw


def metric_g(u, v):
    uu, vv = set(u), set(v)
    u_diff_v = uu - vv
    v_diff_u = vv - uu
    common = uu.intersection(vv)
    a = sum(abs(u.index(ch) + 1) for ch in u_diff_v)
    b = sum(abs(v.index(ch) + 1) for ch in v_diff_u)
    sf = spearmanft(u, v, normalize=False)
    g = 2 * (len(u) - len(common)) * (len(u) + 1) + sf - a - b
    g_max = len(u) * (len(u) + 1)
    return 1 - (g / g_max)


def metric_m(u, v):
    uu, vv = set(u), set(v)
    u_diff_v = uu - vv
    v_diff_u = vv - uu
    common = uu.intersection(vv)
    a = sum(abs(1.0 / (u.index(ch) + 1) - 1 / 11.0) for ch in u_diff_v)
    b = sum(abs(1.0 / (v.index(ch) + 1) - 1 / 11.0) for ch in v_diff_u)
    c = sum(abs(1.0 / (u.index(ch) + 1) - 1.0 / (v.index(ch) + 1))
            for ch in common)
    m = c + a + b
    m_max = 2 * sum(abs(1.0 / (i + 1) - 1 / 11.0) for i in range(len(u)))
    return 1 - (m / m_max)


def kendalltau_distance(u, v):
    uu, vv = set(u), set(v)
    common = uu.intersection(vv)
    pairs = combinations(common, 2)
    distance = 0.0
    for x, y in pairs:
        a = u.index(x) - u.index(y)
        b = v.index(x) - v.index(y)
        if a * b < 0:
            distance += 1
    return 1.0 if len(common) <= 1 else\
        1 - distance / (len(common) * (len(common) - 1) / 2)


def spearmanft(u, v, normalize=True):
    uu, vv = set(u), set(v)
    common = uu.intersection(vv)
    reranked_a = [x for x in u if x in common]
    reranked_b = [x for x in v if x in common]
    sf = sum(abs((reranked_a.index(ch) + 1.0) - (reranked_b.index(ch) + 1.0))
             for ch in common)

    def normalize_sf():
        iseven = len(common) % 2 == 0
        maxvalue = (len(common) ** 2) / 2 if iseven else \
            (len(common) - 1) * (len(common) + 1) / 2

        value = 0 if len(common) <= 1 else sf / maxvalue
        assert 0 <= value <= 1
        return value

    return normalize_sf() if normalize else sf


def _transpositions_penalty(u, v):
    assert len(u) == len(v)
    N = len(u)
    uu, vv = set(u), set(v)
    common = uu & vv
    total = sum(abs(u.index(x) - v.index(x)) for x in common)
    maxvalue = sum(N - i - 1 if not i % 2 else N - i
                   for i in range(len(common)))
    value = float(total) / maxvalue
    assert 0 <= value <= 1
    return value


def _extract_item_from_dict(dict_value):
    if len(dict_value) != 1:
        raise utils.SEAnalysisException(
            "The length of the given dictionary is not equal to 1")

    return (
        next(iter(dict_value.keys())),
        next(iter(dict_value.values()))
    )


def metric_T(u, v, snippets_u, snippets_v, titles_u, titles_v, a, b, c,
             W=[0.15, 0.1, 0.07, 0.04, 0.01], threshold=0.1):
    u_len, v_len = len(u), len(v)
    if not u_len or not v_len:
        return 0

    if W and any(w_a < w_b for w_a, w_b in zip(W, W[1:])):
        raise utils.SEAnalysisException(
            'Elements of `W` must have a descending order')
    common = (set(u) - {utils.NO_RESULT}) & (set(v) - {utils.NO_RESULT})
    m = float(len(common))

    if not m:
        return 0

    k = 0
    s = 0

    title_u_key, title_u_val = _extract_item_from_dict(titles_u)
    title_v_key, title_v_val = _extract_item_from_dict(titles_v)

    snippet_u_key, snippet_u_val = _extract_item_from_dict(snippets_u)
    snippet_v_key, snippet_v_val = _extract_item_from_dict(snippets_v)

    for ch in common:
        x = u.index(ch)
        y = v.index(ch)

        title1 = title_u_val[x]
        if type(title1) != str:
            title1 = title1.decode('utf-8')

        title2 = title_v_val[y]
        if type(title2) != str:
            title2 = title2.decode('utf-8')

        titles = {
            'u': {
                title_u_key: title1
            },
            'v': {
                title_v_key: title2
            }
        }
        bows = bw.BagOfWords(titles, False).build_bows()
        k += (1 - cosine(bows[0].matrix, bows[1].matrix).flatten()[0])


        snippet1 = snippet_u_val[x]
        if type(snippet1) != str:
            snippet1 = snippet1.decode('utf-8')

        snippet2 = snippet_v_val[y]
        if type(snippet2) != str:
            snippet2 = snippet2.decode('utf-8')

        snippets = {
            'u': {
                snippet_u_key: snippet1
            },
            'v': {
                snippet_v_key: snippet2
            }
        }
        bows = bw.BagOfWords(snippets).build_bows()
        s += (1 - cosine(bows[0].matrix, bows[1].matrix).flatten()[0])

    tr = _transpositions_penalty(u, v)

    weight = np.divide(np.subtract(np.subtract(
        np.subtract(3.0 * m + 1, np.multiply(a, s)),
        np.multiply(b, k)), np.multiply(c, tr)), (3.0 * len(u) + 1))

    for k in range(len(weight)):
        if W and weight[k] >= threshold:
            j = min(u_len, len(W))
            i = 0
            total = 0
            while i < j:
                if u[i] == v[i]:
                    total += W[i]
                i += 1
            if total:
                weight[k] += total * (1.0 - weight[k])

    assert not np.any(weight < 0)

    return weight

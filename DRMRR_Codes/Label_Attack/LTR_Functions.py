"""
*************************************************************************
*************************************************************************
    Python implementation of DRMRR algorithm introduced in the paper: 
 "Distributionally robust learning-to-rank under the Wasserstein metric" 
     Shahabeddin Sotudian, Ruidi Chen, and Ioannis Ch. Paschalidis
*************************************************************************
*************************************************************************
"""


import numpy as np
import six


def Read_Line(lines, has_targets=True, one_indexed=True, missing=0.0):
    for line in lines:
                   
        data, _, comment = line.rstrip().partition('#')
        toks = data.split()

        num_features = 0
        x = np.repeat(missing, 8)
        y = -1.0
        if has_targets:
            y = float(toks[0])
            toks = toks[1:]

        qid = _parse_qid_tok(toks[0])

        for tok in toks[1:]:
            fid, _, val = tok.partition(':')
            fid = int(fid)
            val = float(val)
            if one_indexed:
                fid -= 1
            assert fid >= 0
            while len(x) <= fid:
                orig = len(x)
                x.resize(len(x) * 2)
                x[orig:orig * 2] = missing

            x[fid] = val
            num_features = max(fid + 1, num_features)

        assert num_features > 0
        x.resize(num_features)

        yield (x, y, qid, comment)


   
def Read_LTR_Dataset(source, has_targets=True, one_indexed=True, missing=0.0):
    if isinstance(source, six.string_types):
        source = source.splitlines()

    max_width = 0
    xs, ys, qids, comments = [], [], [], []
    it = Read_Line(source, has_targets=has_targets,
                    one_indexed=one_indexed, missing=missing)
    for x, y, qid, comment in it:
        xs.append(x)
        ys.append(y)
        qids.append(qid)
        comments.append(comment)
        max_width = max(max_width, len(x))

    assert max_width > 0
    X = np.ndarray((len(xs), max_width), dtype=np.float64)
    X.fill(missing)
    for i, x in enumerate(xs):
        X[i, :len(x)] = x
    ys = np.array(ys) if has_targets else None
    qids = np.array(qids)
    comments = np.array(comments)

    return (X, ys, qids, comments)
   

def _parse_qid_tok(tok):
    assert tok.startswith('qid:')
    return tok[4:]


def Precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if len(r) != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def AP_K(r,K):
    r = np.asarray(r) != 0
    out = [Precision_at_k(r, i + 1) for i in range(K) if r[i]]
    if not out:
        return 0.
    return np.mean(out)

def MRR(RGB):
    Non_Zeros = (np.asarray(RGB).nonzero()[0])
    if Non_Zeros.size == 0:
        return 0.
    else:
        return 1. / (Non_Zeros[0] + 1)
    
  
def DCG_K(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def NDCG_K(r, k):
    dcg_max = DCG_K(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return DCG_K(r, k) / dcg_max

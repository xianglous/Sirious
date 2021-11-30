import numpy as np
from scipy import spatial
from sentence_transformers import SentenceTransformer
from nltk import tokenize
from transformers import GPT2TokenizerFast

from util import *


class DocCutter:
    """
    A utility to break down a long text into sequential, disjoint segments based on semantic similarity.

    Each segment contains complete sentences.
    """
    def __init__(self):
        self.d_models = {}

    def __call__(self, txt: str, method='all-MiniLM-L6-v2', max_sz=2**8, n_counter=None):
        """
        :param txt: Text to cut into segments
        :param method: A sentence transformer name
        :param max_sz: Maximum size (inclusive) allowed for each segment
        :param n_counter: Function that returns the number of tokens given a text segment, or a list of text segments
            Default to length of GPT-2 tokenization
        """
        if method not in self.d_models:
            self.d_models[method] = SentenceTransformer(method)
        if n_counter is None:
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

            def counter(x):
                ids = tokenizer(x)['input_ids']
                if isinstance(x, str):
                    return len(ids)
                else:  # list[str]
                    return sum(len(e) for e in tokenizer(ids))
            n_counter = counter
        sents = tokenize.sent_tokenize(txt)
        model = self.d_models[method]
        vec_sents = np.vstack([model.encode(sent) for sent in sents])
        n = len(sents)
        # sims_vec = np.empty((n, n), vec_sents.dtype)  # Similarity score for each pair
        # sims_vec = {
        #     (i, j): spatial.distance.cosine(vec_sents[i], vec_sents[j]) for i in range(n-1) for j in range(i+1, n)
        # }
        # ic(sims_vec)

        ic(type(vec_sents), vec_sents.shape, vec_sents[0].shape)
        # exit(1)

        def split(chunk: list[int]):
            """
            Break down `chunk` into half, by greedily picking the split point with lowest cos similarity
            between the two segments, recursively, until each segment has words less than max_sz

            :param chunk: Continuous array indices, each corresponding to a sentence in `sent`
            """
            def _vect(chunk_):
                """
                Results previously computed are cached
                """
                if not hasattr(_vect, 'memo'):
                    _vect.memo = {}
                k = chunk_[0], chunk_[-1]
                if k not in _vect.memo:
                    # ic(vec_sents[chunk_].shape)
                    _vect.memo[k] = np.mean(vec_sents[chunk_], axis=0) / len(chunk_)  # Normalize by sentence count
                return _vect.memo[k]

            def _sim(i, j):
                assert 0 <= i < n and 0 <= j < n and i < j
                if not hasattr(_vect, 'memo'):
                    _vect.memo = {}
                k = (i, j)
                if k not in _vect.memo:
                    _vect.memo[k] = spatial.distance.cosine(vec_sents[i], vec_sents[j])
                return _vect.memo[k]
                # vecs1 = vec_sents[chunk[:idx]]
                # vecs2 = vec_sents[chunk[idx:]]
                # ic(vecs1.shape[0], vecs2.shape[0])
                # vecs_l, vecs_s = (vecs1, vecs2) if vecs1.shape[0] > vecs2.shape[0] else (vecs2, vecs1)
                # return

            def sim(idx):
                c1 = chunk[:idx]
                c2 = chunk[idx:]
                # ic(c1, c2)
                assert len(c1) >= 1 and len(c2) >= 1
                sims = np.array([_sim(i, j) for i in c1 for j in c2])
                # return sum() / (len(c1) * len(c2))
                # ic(sims, sims.mean(), np.min(sims), np.max(sims))
                # return np.min(sims)
                return sims.mean()

            # ic(np.diff(chunk))
            np.testing.assert_array_equal(np.array(chunk), np.arange(chunk[0], chunk[-1]+1))
            # if sum(n_counter(sents[idx]) for idx in chunk) <= max_sz:
            if sum(n_counter(sents[idx]) for idx in chunk) <= max_sz:
                return chunk
            else:
                # sim_scores = {  # Higher cosine similarity = Lower cosine distance
                #     i: spatial.distance.cosine(_vect(chunk[:i]), _vect(chunk[i:])) for i in range(1, len(chunk))
                # }
                # idx = max(sim_scores, key=sim_scores.get)  # Pick split point with smallest similarity
                # ic(sim_scores, idx)
                # ic(_vect([1, 2, 3]))
                sim_scores = {idx: sim(idx) for idx in range(1, len(chunk))}
                idx = max(sim_scores, key=sim_scores.get)  # Pick split point with smallest similarity
                # ic(sim_scores, idx)
                return split(chunk[:idx]), split(chunk[idx:])

        return split(list(range(len(sents))))


if __name__ == '__main__':
    from icecream import ic

    dc = DocCutter()
    t = get_ted_eg('Cuddy')['transcript']
    t = ' '.join(tokenize.sent_tokenize(t)[:15])  # Get a smaller sample
    # t = get_498_eg()
    # ic(t[:400])
    ic(tokenize.sent_tokenize(t))
    ic(dc(t, max_sz=128))

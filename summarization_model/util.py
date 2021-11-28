import re
import glob
import json

from icecream import ic


def is_mmss(s):
    """
    :return: True if input string follow the time format `mm:ss`
    """
    return re.match(r'^\d\d:\d\d$', s)


def get_1st_n_words(text, n=None, as_arr=False):
    words = text.split()[:n]
    return words if as_arr else ' '.join(words)


def load_ted_clean(fl):
    """
    :param fl: An opened file
    :return: Array of strings, The complete speech text
    """
    def _content_relevant(x):
        return not (x.startswith('(Laughter)') or is_mmss(x))

    lns = list(map(str.strip, fl.readlines()))
    lns = list(filter(str.strip, lns))
    lns = list(filter(bool, lns))
    lns = list(filter(_content_relevant, lns))
    lns = [ln.removeprefix('(Audience)') for ln in lns]
    return lns


def get_ted_eg(k='Do schools kill creativity'):
    """
    Talks in ~20min comes with ~3k tokens, already don't fit in most traditional models
    
    :return: `dict` about a TED talk
    """
    # fnm = 'data/example/ted_does-schools-kill-creativity, cleaned.txt'
    # with open(fnm) as f:
    #     return ' '.join(load_ted_clean(f))
    if not hasattr(get_ted_eg, 'dset'):
        with open(f'dataset/ted-summaries.json') as f:
            get_ted_eg.dset = json.load(f)
    if k:
        if type(k) is int:
            return get_ted_eg.dset[k]
        else:  # Expect str
            # ic(list(filter(lambda d: k in d['title'], get_ted_eg.dset)))
            return next(filter(lambda d: k in d['title'], get_ted_eg.dset))
    else:
        return get_ted_eg.dset


def get_498_eg(section=False, cleaned=True):
    d = '../Transcription/transcripts'
    fnm = f'eecs498_lec03{", cleaned" if cleaned else ""}{", section" if section else ""}'
    fnm = f'{d}/{fnm}.txt'
    f = open(fnm)
    lns = list(map(str.strip, f.readlines()))
    return ' '.join(lns)


if __name__ == '__main__':
    # fnms = glob.glob('**/*.txt', recursive=True)
    # ic(fnms)
    # fnm = list(filter(lambda x: 'cleaned' in x, fnms))[0]
    # ic(fnm)
    # with open(fnm) as f:
    #     txts = load_ted_clean(f)
    #     ic(len(txts))
    #
    #     txt = ' '.join(txts)
    #     n_words = len(txt.split())
    #     ic(n_words)
    #     ic(txt[:500])
    #     # ic(txt)
    ic(get_ted_eg())

    txt = get_498_eg()
    ic(len(txt.split()))

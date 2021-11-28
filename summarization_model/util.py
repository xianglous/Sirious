import re
import json
from typing import Union

from icecream import ic


def is_mmss(s):
    """
    :return: True if input string follow the time format `mm:ss`
    """
    return re.match(r'^\d\d:\d\d$', s)


def get_1st_n_words(text, n=None, as_arr=False):
    words = text.split()[:n]
    return words if as_arr else ' '.join(words)


# def load_ted_clean(fl):
#     """
#     :param fl: An opened file
#     :return: Array of strings, The complete speech text
#     """
#     def _content_relevant(x):
#         return not (x.startswith('(Laughter)') or is_mmss(x))
#
#     lns = list(map(str.strip, fl.readlines()))
#     lns = list(filter(str.strip, lns))
#     lns = list(filter(bool, lns))
#     lns = list(filter(_content_relevant, lns))
#     lns = [ln.removeprefix('(Audience)') for ln in lns]
#     return lns

def clean_ted(ted: Union[dict, str]):
    def _clean_ted(s):
        return re.sub(r'\((Laughter|Audience|Applause)\)', '', s)
    if isinstance(ted, dict):
        ted['transcript'] = _clean_ted(ted['transcript'])
        return ted
    else:
        assert isinstance(ted, str)
        return _clean_ted(ted)


def get_ted_eg(k='Do schools kill creativity', clean=True):
    """
    Talks in ~20min comes with ~3k tokens, already don't fit in most traditional models
    
    :return: `dict` about a TED talk
    """
    if not hasattr(get_ted_eg, 'dset'):
        with open(f'dataset/ted-summaries.json') as f:
            get_ted_eg.dset = json.load(f)

    def _ted_eg():
        if k:
            if k == 'Cuddy':
                with open('data-eg/Amy Cuddy: Your body language may shape who you are.json') as f:
                    return json.load(f)
            elif type(k) is int:
                return get_ted_eg.dset[k]
            else:  # Expect str
                return next(filter(lambda d: k in d['title'], get_ted_eg.dset))
        else:
            return get_ted_eg.dset
    ret = _ted_eg()
    # ic(ret)
    return (
        (clean and isinstance(ret, list) and [clean_ted(t) for t in ret]) or
        (clean and isinstance(ret, dict) and clean_ted(ret)) or
        (not clean and ret)
    )


def get_498_eg(section=False):
    d = '../Transcription/transcripts'
    fnm = 'eecs498_lec03'
    if section:
        fnm = f'{fnm}, section'
    f = open(f'{d}/{fnm}.txt', 'r')
    lns = list(map(str.strip, f.readlines()))
    return ' '.join(lns)


if __name__ == '__main__':
    # ic(get_ted_eg())
    print(get_ted_eg('Cuddy')['transcript'])

    txt = get_498_eg()
    ic(len(txt.split()))

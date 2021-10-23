import re
import glob
from icecream import ic


def is_mmss(s):
    """
    :return: True if input string follow the time format `mm:ss`
    """
    return re.match(r'^\d\d:\d\d$', s)


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


if __name__ == '__main__':
    fnms = glob.glob('**/*.txt', recursive=True)
    fnm = list(filter(lambda x: 'cleaned' in x, fnms))[0]
    with open(fnm) as f:
        txts = load_ted_clean(f)
        ic(len(txts))

        txt = ' '.join(txts)
        ic(txt[:500])

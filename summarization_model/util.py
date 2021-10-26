import re
import glob
from icecream import ic


def is_mmss(s):
    """
    :return: True if input string follow the time format `mm:ss`
    """
    return re.match(r'^\d\d:\d\d$', s)


def get_1st_n_words(text, n=2**10):
    words = text.split()
    return ' '.join(words[:n])


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


def get_ted_eg(crop=True):
    """
    3310 words in the text, doesn't fit in most traditional models

    [Does School Kill Creativity?](https://www.ted.com/talks/sir_ken_robinson_do_schools_kill_creativity)
    by Sir Ken Robinson
    """
    FNM = 'data/example/ted_does-schools-kill-creativity, cleaned.txt'
    with open(FNM) as f:
        return ' '.join(load_ted_clean(f))


def get_498_eg(section=True):
    fnm = 'eecs498_lec03, section' if section else 'example'
    fnm = f'../Transcription/transcripts/{fnm}.txt'
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

    txt = get_498_eg()
    ic(len(txt.split()))

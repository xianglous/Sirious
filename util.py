from rpunct import RestorePuncts
from icecream import ic


rp = RestorePuncts(use_cuda=False)


def restore_punctuation(text):
    return rp.punctuate(text)


if __name__ == '__main__':
    # txt = 'in 2018 cornell researchers built a high-powered detector that in combination with an algorithm-driven ' \
    #       'process called ptychography set a world record by tripling the resolution of a state-of-the-art electron ' \
    #       'microscope'
    txt = 'in 2018 cornell researchers built a high-powered detector'
    ic(restore_punctuation(txt))  # Doesn't terminate

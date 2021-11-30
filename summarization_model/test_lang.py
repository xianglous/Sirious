from icecream import ic
from util import *


if __name__ == '__main__':
    # import re
    # t = '18:26'
    # t_n1 = '18:26a'
    # t_n2 = '18:261'
    # t_n3 = '18:2'
    #
    # for i in [t, t_n1, t_n2, t_n3]:
    #     m = re.match(r'^\d\d:\d\d$', i)
    #     ic(i, m)

    # import torch
    # x = torch.rand(5, 3)
    # ic(x)

    # import nltk
    # nltk.download('punkt')

    from nltk import tokenize
    from transformers import GPT2TokenizerFast
    from nltk.tokenize import RegexpTokenizer
    # tokenizer = RegexpTokenizer(r'\w+')
    # text = "This is my text. It icludes commas, question marks? and other stuff. Also U.S.."
    # ic(tokenizer.tokenize(text))

    txt = 'There\'s a passage that I got memorized, seems appropiate for this situation: Ezekiel 25,17. "The path of ' \
          'the righteous man is beset of all sides by the iniquities of the selfish and the tyranny of evil me. ' \
          'Blessed is he who, in the name of the charity and good will, shepherds the weak through the valley of ' \
          'darkness, for he is truly his brother\'s keeper and the finder of lost children. \nAnd I will strike down ' \
          'upon thee with great vengeance and furious anger those who attempt to poison and destroy my brothers. And ' \
          'you will know my name is the Lord when I lay my vengeance upon thee. '
    ic(txt)
    ic(tokenize.sent_tokenize(txt))
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    ic(tokenizer("Hello world")['input_ids'])
    ic(tokenizer(" Hello world")['input_ids'])
    tokens = tokenizer(txt)
    # ic(tokens, type(tokens))
    ic(len(tokens['input_ids']), len(tokenizer.tokenize(txt)))
    lst = tokenizer([
        'sentence 1',
        'another one',
        'what does it return?'
    ])['input_ids']
    ic(lst, type(lst))

    # arr = [26, 27, 28, 29, 30, 31]
    # for i in range(1, len(arr)):
    #     ic(i, arr[:i], arr[i:])

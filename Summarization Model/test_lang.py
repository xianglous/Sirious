from icecream import ic


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

    import torch
    x = torch.rand(5, 3)
    ic(x)

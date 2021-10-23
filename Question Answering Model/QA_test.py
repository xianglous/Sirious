from allennlp_models.pretrained import load_predictor
predictor = load_predictor("coref-spanbert")


def resolve(lecture):
    with open(f'context/Lecture_{lecture}.txt', 'r', encoding='utf-8') as fin:
        context = fin.read()
    conts = []
    for cont in context.split('\n\n\n'):
        print('Working on chunk', len(conts))
        conts.append(predictor.coref_resolved(cont))
    with open(f'context/Lecture_{lecture}_allen_resolved.txt', 'w+', encoding='utf-8') as fout:
        fout.write('\n\n\n'.join(conts))


def main():
    resolve(2)
    resolve(3)


if __name__ == '__main__':
    main()
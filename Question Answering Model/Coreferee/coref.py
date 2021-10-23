import spacy
import coreferee

# nlp = spacy.load("en")
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('coreferee')


def resolve(text):
    doc = nlp(text)
    new_doc = [token.text_with_ws for token in doc]
    for chain in doc._.coref_chains:
        for mention in chain:
            for index in mention:
                # print(index, new_doc[index])
                resolved = doc._.coref_chains.resolve(doc[index])
                if resolved:
                    new_doc[index] = 'and '.join([token.text_with_ws for token in resolved])
                # print(new_doc[index])
    return ''.join([token for token in new_doc])


def main():
    resolved = resolve('Deepika has a dog. She loves him. The movie star has always been fond of animals')
    # print(clusters)
    print(resolved)
    
    # print(bert_out('Deepika has a dog. She loves him. The movie star has always been fond of animals'))


if __name__ == '__main__':
    main()
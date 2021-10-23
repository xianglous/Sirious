# import spacy
# import stanza
# stanza.install_corenlp()
# stanza.download_corenlp_models(model='english', version='4.2.2')
from stanza.server import CoreNLPClient
# import os
# os.environ["CORENLP_HOME"] = '~/stanza_corenlp'
# nlp = stanza.Pipeline()

'''
nlp = spacy.load("en")
neuralcoref.add_to_pipe(nlp)

def resolve(text):
    doc = nlp(text)
    return doc._.coref_resolved
'''
def resolve(client, text):
    
    with CoreNLPClient(
            properties={'annotators': 'tokenize, ssplit, pos, lemma, ner, parse, coref', 
                        'coref.algorithm': 'clustering'},
            timeout=10000000,
            memory='16G') as client:
        ann = client.annotate(text)
        # print(dir(ann.sentence[0].token))
        orig_ann = [[token.word for token in sentence.token] for sentence in ann.sentence]
        copy_ann = [[token.word for token in sentence.token] for sentence in ann.sentence]
        coref_chain = ann.corefChain
        # print(coref_chain)
        for chain in coref_chain:
            mychain = []
            # Loop through every mention of this chain
            first_mention = chain.mention[0]
            replace = ann.sentence[first_mention.sentenceIndex].token[first_mention.beginIndex:first_mention.endIndex]
            for mention in chain.mention:
                # Get the sentence in which this mention is located, and get the words which are part of this mention
                # (we can have more than one word, for example, a mention can be a pronoun like "he", but also a compound noun like "His wife Michelle")
                if mention.mentionType == 'NOMINAL':
                    replace = ann.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex]
                    break
            for mention in chain.mention:
                copy_ann[mention.sentenceIndex][mention.beginIndex] = '_'.join([token.word for token in replace])
                for index in range(mention.beginIndex + 1, mention.endIndex):
                    copy_ann[mention.sentenceIndex][index] = ''
                if ann.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex] == replace:
                    orig_ann[mention.sentenceIndex][mention.beginIndex] = '_'.join([token.word for token in replace])
                    for index in range(mention.beginIndex + 1, mention.endIndex):
                        orig_ann[mention.sentenceIndex][index] = ''
            
        orig_sentence = ' '.join([' '.join([word for word in words if word != '']) for words in orig_ann])
        new_sentence = ' '.join([' '.join([word for word in words if word != '']) for words in copy_ann])
        return orig_sentence, new_sentence



def main():
    
    text = 'My sister has a dog. She loves him.'

    resolve(text)
    '''
    resolved = resolve('Deepika has a dog. Deepika loves dog. The movie star has always been fond of animals')
    # print(clusters)
    print(resolved)
    '''
    # print(bert_out('Deepika has a dog. She loves him. The movie star has always been fond of animals'))


if __name__ == '__main__':
    # pass
    main()
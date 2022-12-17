'''
TODO
'''

import io
import fasttext
import numpy as np

fasttext.FastText.eprint = lambda x: None # Silence load_model warnings

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}

    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])

    return data

data = load_vectors('wiki_news.vec')
print(data)









model = fasttext.load_model('wiki_news.vec')
my_document = 'hello world, my name is rex askjhkhbb.'

word_embs = np.array([model.get_word_vector(word) for word in my_document.split()])

print(fasttext.tokenize(my_document))

print(model.labels)
for i in word_embs:
    print(i)

'''
Clean and tokenize sentences
Get a document embedding
Optionally convert text to an image
'''
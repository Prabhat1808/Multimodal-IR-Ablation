from gensim.models import Word2Vec
import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import nltk
import json

#path to xml files
xml_dirpath = '/mnt/f/mtp/dataset/dataset/wiki_top10cats/wikipedia_dataset/texts/'
xml_files = [join(xml_dirpath, f) for f in listdir(xml_dirpath) if isfile(join(xml_dirpath, f))]

# tokenizer to remove punctuation marks
tokenizer = nltk.RegexpTokenizer(r"\w+")

wiki_classes = ['art', 'biology', 'geography', 'history', 'literature', 'media', 'music', 'royalty', 'sport', 'warfare']

def getTextContent(filepath):
    """
        Given the name of the xml file, it returns content of 
        <text> </text> field
    """
    with open(filepath, 'r') as file_obj:
        text = file_obj.read()
    content = text.split('text>')[1]
    return content

def preprocess(text):
    """ preprocess text """
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    return tokens

def fetchDict(filepath):
    """ read dict from given filepath """
    with open(filepath, 'r') as file_obj:
        dict_obj = json.load(file_obj)
    return dict_obj

def dumpToJson(dt, filename):
    """ dump dict dt with filename on disk"""
    with open(filename, 'w') as f_obj:
        json.dump(dt, f_obj)

# preparing the training data for word2vec model
print('preparing training data...')
train_data = []
for filepath in xml_files:
    train_data.append(preprocess(getTextContent(filepath)))

# training word2vec
print('training the word2vec model...')
model = Word2Vec(train_data, vector_size=4096, window=4, min_count=1, workers=4)

#getting the embeddings
class_embedding = { cl : model.wv[cl].tolist() for cl in wiki_classes }

dumpToJson(class_embedding, 'wiki_class_embedding.json')















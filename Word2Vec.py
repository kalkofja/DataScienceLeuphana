import gensim
import gzip
import logging
import os
from gensim.models import KeyedVectors


def read_input(input_file):
    logging.info("reading file {0}... this may take a while".format(input_file))
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            if (i % 1000 == 0):
                print(("read {0} lines".format(i)))
            yield gensim.utils.simple_preprocess(line)


abspath = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(abspath, "./reviews_data.txt.gz")

documents = list(read_input(data_file))
logging.info("Done reading data file")

model = gensim.models.Word2Vec(
    documents,
    window=5,
    min_count=2,
    workers=8)

w1 = "dirty"
print(model.wv.most_similar(positive=w1))
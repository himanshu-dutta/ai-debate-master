#!/usr/bin/env python
# coding: utf-8

# # Dependencies

import numpy as np
import pickle
import re
import nltk
import gensim
import multiprocessing
from nltk.tokenize import word_tokenize



import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



setting = {
            'embedding_size' : 100,
            'min_count' : 2,
            'epochs' : 100
        }



# # Model


def tag_documents(word_tokens,label=True):
    tagged_sentences = []
    for i in range(len(word_tokens)):
        tagged_sentences.append(gensim.models.doc2vec.TaggedDocument(word_tokens[i], [i]))
    return tagged_sentences
    


def get_ranks(tagged_sentences,model):
    ranks = []
    for doc_id in range(len(tagged_sentences)):
        print(doc_id)
        inferred_vector = model.infer_vector(tagged_sentences[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
    return ranks



def create_model(setting):
    cores = multiprocessing.cpu_count()
    return gensim.models.doc2vec.Doc2Vec(vector_size=setting['embedding_size'], min_count=setting['min_count'], epochs=setting['epochs'],workers=cores, alpha=0.025, min_alpha=0.001)


def get_dict(model, tagged_sentences):
    return model.build_vocab(tagged_sentences,progress_per=50000)



def train_model(model,tagged_sentences):
    return model.train(tagged_sentences, total_examples=model.corpus_count, epochs=model.epochs)



def get_similar_sentences(query,model,tagged_sentences,top_n=10):
    inferred_vector = model.infer_vector(word_tokenize(query))
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    similar_docs=[]
    for index in range(len(sims[:top_n])):
        similar_docs.append(' '.join(tagged_sentences[sims[index][0]].words))
    return similar_docs


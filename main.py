#!/usr/bin/env python
# coding: utf-8

import Scrapper
import Preprocessing
import ContextAnalysis
import SentimentAnalysis
import time
import os
import gensim
import multiprocessing
from collections import OrderedDict
from random import shuffle
from gensim.models.doc2vec import Doc2Vec
import pickle
import pprint
from flask import Flask, render_template,request, session
from sklearn import utils
# session.clear()
# eel.init('web')
app = Flask(__name__)

#Basic Pickle Functions
def save_data(name, data):
    pickle.dump(data, open( name+".p", "wb" ))
def load_data(loc):
    return pickle.load( open( loc, "rb" ))

# global files required
start = topic = links = articles = corpus = sent_tokens = word_tokens = models = tagged_sentences = None; 




@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")


path_links = 'Data/Links/'
path_articles = 'Data/Articles/'


@app.route('/prepare/',methods=['GET','POST'])
def prepare():
    global start, topic, links, articles, corpus, sent_tokens, word_tokens, models, tagged_sentences
    start = time.time()
    topic = [x for x in request.form.values()][0]
    print(topic)
    links = Scrapper.get_links(topic,100)
    save_data(path_links + topic,links)
    # # topic = 'Abortion'
    # links = load_data('Data/Links/'+topic+'.p')
    

    articles = Scrapper.get_articles(links)
    save_data(path_articles+topic,articles)
    # articles = load_data('Data/Articles/'+topic+'.p')
    
    corpus, sent_tokens, word_tokens = Preprocessing.data_preprocessing(articles)

    print("Time taken, checkpoint-1:{} mins.".format((time.time()-start)/60))

    tagged_sentences = ContextAnalysis.tag_documents(word_tokens)


    # ContextAnalysis.train_model(model,tagged_sentences)
    # model.train(utils.shuffle(tagged_sentences), total_examples=model.corpus_count, epochs=model.epochs)

    # for epoch in range(50):
    #     model.train(utils.shuffle(tagged_sentences), total_examples = len(tagged_sentences),epochs=1)
    #     model.alpha -= 0.002
    #     model.min_alpha = model.alpha


    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

    # #context model parameters
    common_kwargs = dict(
        vector_size=100, epochs=100, min_count=2,
        sample=0, workers=multiprocessing.cpu_count(), negative=5, hs=0,
    )


    models = [Doc2Vec(dm=0, **common_kwargs)]



    

    
    # #context model structure

    # models = [
    #     # PV-DBOW plain
    #     Doc2Vec(dm=0, **common_kwargs),
    #     # PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
    #     Doc2Vec(dm=1, window=10, alpha=0.05, comment='alpha=0.05', **common_kwargs),
    #     # PV-DM w/ concatenation - big, slow, experimental mode
    #     # window=5 (both sides) approximates paper's apparent 10-word total window size
    #     Doc2Vec(dm=1, dm_concat=1, window=5, **common_kwargs),
    # ]

    #vocabulary building
    print('Building vocaublary.')
    for model in models:
        model.build_vocab(tagged_sentences)
        print("%s vocabulary scanned & state initialized" % model)

    #training
    print('Training the model.')
    models_by_name = OrderedDict((str(model), model) for model in models)

    shuffled_alldocs = tagged_sentences[:]
    shuffle(shuffled_alldocs)

    for model in models:
        print("Training %s" % model)
        model.train(shuffled_alldocs, total_examples=len(shuffled_alldocs), epochs=model.epochs)

    print("Time taken, checkpoint-2:{} mins.".format((time.time()-start)/60))
    return ('', 204)

@app.route('/argue/',methods=['GET','POST'])
def argue():
    query = [x for x in request.form.values()][0]
    print(query)
    global start, topic, links, articles, corpus, sent_tokens, word_tokens, models, tagged_sentences
    in_corpus, in_sent_tokens, in_word_tokens = Preprocessing.data_preprocessing(query)


    #Get sentences of same context as the input query.
    same_context = []
    for i in in_sent_tokens:
        for model in models:
            same_context += ContextAnalysis.get_similar_sentences(i,model,tagged_sentences,top_n=5)
        



    #Get the sentences of the same and reasonable polarity.
    sent_score = SentimentAnalysis.get_sentence_polarity(same_context)

    out = []
    for k in range(len(in_sent_tokens)):
        out.append(SentimentAnalysis.find_sentences(in_sent_tokens[k],same_context,sent_score,similar=True,top_n=10)[0])
        out.append(SentimentAnalysis.find_sentences(in_sent_tokens[k],same_context,sent_score,similar=True,top_n=10)[1])
    #output processing
    out = list(set(out))

    print('Input:\n',query)
    print('Output:\n','.'.join(out))

    print("Time taken Checkpoint-3:{} mins.".format((time.time()-start)/60))
    return render_template('index.html', argument='What I have to say is....\n {}'.format('.'.join(out)))

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5080))
	#run the app locally on the givn port
	# app.run(host='127.0.0.1', port=port)
	#optional if we want to run in debugging mode
	app.run(debug=True)

# eel.start('index.html', size=(1000, 600))

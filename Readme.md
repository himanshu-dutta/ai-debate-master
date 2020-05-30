By the sound of it, it would seem similar to IBM’s Project Debate, but the implementation differs vastly in it’s nature. The concept remains the same, to put a machine against a human and let them debate on a given topic. This project was implemented in Python.

The basic pipeline is as follows:

It would have a Speech-To-Text component, through which it would hear the opponent’s argument as well as the announced topic before the preparation phase starts.

After the topic is announced, the bot will use a Google Custom Search Engine(CSE) which has some sources pre-defined to gather data on the given topic, such as Wikipedia, news websites, etc. When links to relevant webpages are searched, separate scrapping methods devised for all the pre-defined sources are put to work, gathering all the articles and texts which would form the corpus for the bot to train on.

This corpus is then passed to a Doc2Vec model to get an embedding representation of all the different documents in the corpus.

At the same time, the corpus is used to train a Sentiment Analysis model.

After the preparation phase is over, the continuous process for the bot is to listen to the argument of the opponent, assess the sentiment in terms of positivity and negativity as well as it’s context using the embedding that it learned from the training over Doc2Vec model, and based on that select from the corpus, most appropriate argument to counter the opponent. And using a Text-To-Speech component gives the argument in the form of speech.

Depending on the system components such as Text-To-Speech and Web Scrapping would take fairly less time and wouldn’t impact the overall performance that much, but the Context Analysis and Sentiment Analysis models would highly impact the overall performance of the model. The choice of using a Doc2Vec model, over a Word2Vec or a TF-IDF model, was based on both the factors of speed as well as accuracy.

A Word2Vec model uses one of the two strategies to get to the final word embedding. Skip-gram and Continuous-Bag-of-Words(CBoW) model. Both the models are 2 layered(a hidden embedding layer and an output layer) neural networks. Skip-gram model uses the strategy that, given a certain word it tries to predict nearby words called as context word, and CBoW does the opposite by trying to predict the target word given the context. The output layer uses a softmax activation to generate the probability of a word of being the context/target word for respective models.

A Doc2Vec model uses a similar strategy along with the added knowledge that the next word in a context depends on the document/paragraph that it belongs to. For example, the sentence, “My computer has a lot of \_\_\_\_”, can have the next word as virus/features depending on the entire document in which the sentence exists. So, in addition to the context words, a document token is also passed to the neural network model, and a matrix is maintained for all the documents in the corpus. The document token acts as the missing context to the remaining inputs.

For this project, I used the Gensim library, DM(Distributed Memory) Doc2Vec specifically, which is similar to the CBoW model. The document vectors are obtained by training the neural net on the task of inferring a target word based on the context and a context document.

Later the knowledge of the embedding so formed is used to find the most similar sentences for the argument. Once we have all the chosen sentences, then using the sentiment analysis component the ones representing opposite sentiment to the argument by the opponent are filtered and collected together to be a presentable counter, and are sent for conversion from text to speech and are finally outputted.

The choice of using a Doc2Vec model was based on the fact that it performs better in terms of accuracy, and it saves time in training. With even a small corpus with just 100 documents of about 500 words, which are separated into paragraphs making the corpus of about 1000 documents, it good fairly good results in terms of speed. The best metric to test the model and tune it as per the results was to put it to work and to give it arguments to manually examine its performance. And after making modifications to it, the one problem that still persists is that the output along with the relevant output still has some unwanted text, as well as sentences that are in second and third person.

For the frontend part, to keep the performance fast, Flask framework was used to convert components such as preparation and argument into simple API calls and keep the frontend separate from the rest of the project and model training in the backend.

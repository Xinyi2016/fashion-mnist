import pickle
import gensim
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import numpy as np

def plot_coefficients(classifier, feature_names=None, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    if feature_names is None:
        plt.show()
    else:
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
        plt.show()


def compute_coherence_values(dictionary, corpus, limit, start=2, step=6, texts=None):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        if texts is None:
            print("Starting intrinsic measure...")
            cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
        else:
            print("Getting extrinsic measure...")
            cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(cm.get_coherence())
    return model_list, coherence_values


def get_topic(tok, mode="train", num_topics = 2):
    """
    """
    tok = [r.split(" ") if type(r) is str else r for r in tok]
    if mode=="train":
        dictionary = Dictionary(tok)
        corpus = [dictionary.doc2bow(text) for text in tok]
        pickle.dump(corpus, open('data/corpus.pkl', 'wb'))
        dictionary.save('data/dictionary.gensim')
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = num_topics, id2word=dictionary, passes=20)
        ldamodel.save('data/model{}.gensim'.format(str(num_topics)))
    else:
        dictionary = gensim.corpora.Dictionary.load('data/dictionary.gensim')
        corpus = [dictionary.doc2bow(text) for text in tok]
        ldamodel = gensim.models.ldamodel.LdaModel.load('data/model{}.gensim'.format(str(num_topics)))
    topic_lst = [ldamodel.get_document_topics(c) for c in corpus]
    topic = [max(topic, key=lambda x: x[1])[0] for topic in topic_lst]
    return topic


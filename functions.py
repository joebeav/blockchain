import numpy as np
from collections import *
import time
import json
import os
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import bz2
import nltk
from math import log
import itertools
import pickle


def combinations(input_list, acc=""):
	"""
	Find the combinations for a list. 
	"""

	if not input_list:
		yield acc
		return

	next_val = input_list[0]

	for rest in combinations(input_list[1:], acc):
		yield rest

	acc += next_val

	for rest in combinations(input_list[1:], acc):
		yield rest

def cosine_sim(vec1, vec2):
    """
    Calculate Cosine Similarities
    
    :param vec1: 
    :param vec2: 
    :return: absolute cosine similarity for a word pair
    """

    return abs(np.dot(np.transpose(vec1), vec2) / np.sqrt(np.dot(np.transpose(vec1), vec1) * np.dot(np.transpose(vec2), vec2)))


def read_and_store(year, month):
    """
    Read monthly comments, tokenize them, and regroup the tokens by post. 
    
    :param year: 
    :param month: 
    :return: {key (post): value (comments)}, [noun_verb tokens for this month]
    """

    print("reading")

    now = time.time()
    in_path = "../../../jiaqima/BlockChain/" + year + "-" + month + ".bz2"
    out_path_comments = "intermed/documents/documents-" + year + "-" + month + ".json"
    out_path_nounverb = "intermed/nounVerbs/nounVerbs-" + year + "-" + month + ".json"

    if not os.path.isfile(out_path_nounverb):

        f_in = bz2.BZ2File(in_path).readlines()

        print("loaded " + str(len(f_in)) + " comments")

        tokenizer = RegexpTokenizer(r"\b[\w']+\b")
        comments = [(json.loads(line)['link_id'], tokenizer.tokenize(json.loads(line)['body'].lower())) for line in
                    f_in]

        documents = defaultdict(list)
        tags_of_interest = ['NN', 'NNS', 'NN$', 'VBD', 'VBP', 'VBX', 'VBG', 'VB']
        noun_verb = []

        print("Grouping comments")

        for (k, v) in comments:
            noun_verb += [i[0] for i in nltk.pos_tag(v) if i[1] in tags_of_interest]
            words = [word for word in v]
            documents[k] += words

        nv_list = list(set(noun_verb))

        with open(out_path_comments, 'w') as f:
            json.dump(documents, f)
        f.close()

        with open(out_path_nounverb, 'w') as f:
            json.dump(nv_list, f)
        f.close()

    else:

        documents = json.load(open(out_path_comments))
        nv_list = json.load(open(out_path_nounverb))

    print("File IO took {}s.".format(time.time() - now))

    return documents, nv_list


def calculate_tfidf(documents, year, month):
    """
    
    calculate tfidf values for tokens in a particular month
    
    :param documents: documents for a specific month
    :param year: 
    :param month: 
    :return: {key (token): value (tfidf)}
    """

    now = time.time()
    tfidfPath = "intermed/tfidf/tfidf-{}-{}.json".format(year, month)
    if not os.path.isfile(tfidfPath):

        tf_df = defaultdict(Counter)
        for item in documents.items():
            document = item[1]
            for word in set(document):
                tf_df[word].update(tf=document.count(word), df=1)

        tfidf = {}
        N = len(documents)
        for (k, v) in tf_df.items():
            tfidf[k] = v['tf'] * log(N / v['df'])
        print("Calculating TFIDF took {}s.".format(time.time() - now))
        with open(tfidfPath, 'w') as f:
            json.dump(tfidf, f)
        f.close()

    else:
        tfidf = json.load(open(tfidfPath, 'r'))

    return tfidf


def clean(tfidf_dict, nv_list):
    """
    This function removes words that are not in the noun-verb list from the tf-idf list
    :param tfidf_dict: 
    :param nv_list: 
    :return: 
    """

    now = time.time()
    sorted_list = sorted(tfidf_dict.items(), key=lambda t: -t[1])
    eng_stopwords = stopwords.words('english')
    clean_list = [i for i in sorted_list if (not i[0] in eng_stopwords) and (len(i[0]) > 2) and (i[0] in nv_list)]
    print("Cleaning the TFIDF list took {}s".format(time.time() - now))

    return clean_list


def top_sim(embeddings, wordcodes, wordlist, threshold=.0, percentage=.0, k=.0):
    """
    Calculate similarities and return the most similar words for words in wordlist
    
    This function calculates the cosine similarities for word pairs in list. 
    wordlist is a list of (word, tf-idf) tuples.
    embeddings should be a matrix
    wordcodes map a word to the ith row in the embedding matrix
    """

    now = time.time()
    words = [i[0] for i in wordlist]
    cos_sim = defaultdict(float)
    word_pairs = itertools.combinations(words, 2)
    for pair in word_pairs:
        cos_sim[pair] = cosine_sim(embeddings[wordcodes[pair[0]]], embeddings[wordcodes[pair[1]]])

    # # sort the list by the first word and the similarity value
    # sorted_sims = sorted(cos_sim.items(), key=lambda t: (t[0][0], -t[1]))

    top_pairs = []

    if k:

        # find the k most similar words for each word

        top_k = []
        current_word = ""
        past_word = []

        sorted_sims = sorted(cos_sim.items(), key=lambda t: (t[0][0], -t[1]))

        for i, pair in enumerate(sorted_sims):
            if current_word != pair[0][0]:
                current_word = pair[0][0]
                top_k += sorted_sims[i: i + 5]

        top_k = sorted(top_k, key=lambda t: (t[0][1], -t[1]))

        counter = 0
        current_word = ""
        for pair in top_k:
            if current_word == pair[0][1]:
                counter = counter + 1
            else:
                counter = 0
                current_word = pair[0][1]
            if counter < k:
                top_pairs.append(pair)


    sorted_sims = sorted(cos_sim.items(), key=lambda t: -t[1])

    if threshold:

        # find the pairs with similarity greater than the threshold

        for i, pair in enumerate(sorted_sims):
            if pair[1] > threshold:
                top_pairs.append(pair)

    if percentage:

        # find as much pairs with highest similarities so that the related words cover percentage*100 % of the whole vocabulary

        size = 0
        target_size = int(len(wordlist) * percentage)
        related_words = []
        for i, pair in enumerate(sorted_sims):
            for word in pair[0]:
                if not word in related_words:
                    related_words.append(word)
                if len(related_words) < target_size:
                    top_pairs.append(pair)
                else:
                    break

    print("Calculating top similarities took {}s".format(time.time() - now))

    return top_pairs

def best_split(wordPairs):

    """
    Giving a Graph, return the best community partition
    
    :param Graph: a graph constructed with the most similar word pairs 
    :return: (level of partition that gives the best performance, best performance, best partition)
    """
    from networkx.algorithms import community
    from networkx.algorithms.community.quality import performance, coverage
    import networkx as nx

    Graph = nx.Graph()
    edges = [(pair[0][0], pair[0][1]) for pair in wordPairs]
    edgewidth = [pair[1] * 10 for pair in wordPairs]
    Graph.add_edges_from(edges)

    max_pc = 0
    max_index = None
    best_communities = None
    communities_generator = community.girvan_newman(Graph)
    for i, communities in enumerate(communities_generator):
        p = performance(Graph, communities)
        c = coverage(Graph, communities)
        if 2*p*c/(p+c) > max_pc:
            max_index = i
            max_pc = 2*p*c/(p+c)
            best_communities = communities
    return (max_index, max_pc, best_communities)

def glove(year, month, documents, preloadEmbeddings, preloadW2c):
    """
    Update the GloVe embeddings using the tokens of the current month and embeddings from last month as initialization.
    
    :param year: 
    :param month: 
    :param documents: {post_id : list of tokens}
    :param preloadEmbeddings: embeddigns matrix
    :param preloadW2c: {word : onehot index}
    :return: updated embeddings and word_2_code indices
    """

    import tf_glove

    embPath = "intermed/embeddings/embeddings-{}-{}.p".format(year, month)
    w2cPath = "intermed/w2c/w2c-{}-{}.p".format(year, month)

    if not os.path.isfile(embPath):
        wordlist = []
        for k, v in documents.items():
            wordlist.append(v)

        model = tf_glove.GloVeModel(embedding_size=300, context_size=10, pre_load_weights=preloadEmbeddings,
                                    pre_load_w2c=preloadW2c)
        model.fit_to_corpus(wordlist)
        model.train(num_epochs=100)

        embeddings = model.embeddings
        pickle.dump(embeddings, open(embPath, "wb+"))

        w2c = model.word_to_id()
        pickle.dump(w2c, open(w2cPath, "wb+"))

    else:
        embeddings = pickle.load(open(embPath, 'rb+'))
        w2c = pickle.load(open(w2cPath, 'rb+'))

    return embeddings, w2c


if __name__ == "__main__":

    print ("Utility functions.")

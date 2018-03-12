from functions import *
import json


def detect_communities(year, month, preloadEmb = None, preloadW2c = None):
    """
    
    :param year: the year of the data to manipulate
    :param month: the month of the data to manipulate
    :return: updated embeddings, w2c, communities for this month
    """

    """ load documents """
    documents, nvList = read_and_store(year, month)
    documents = documents[:100] # using a small portion for test runs, comment this line for real runs

    """ calculate tf-idf """
    tfIdf_dict = calculate_tfidf(documents, year, month)

    """ keep only the relevant terms """
    tfIdf_clean = clean(tfIdf_dict, nvList)[:100] # using only top100 for test runs, change to 1000 for real runs

    """ load trained embeddings """
    trainedEmb, word2code = glove(year, month, documents, preloadEmb, preloadW2c)

    # """ find the best threshold """
    # thresholdOPT = 0
    # perfMax = 0
    # communityOPT = 0
    # perf = open("results/k-{}-{}_signed_cos.txt".format(year, month), "w")
    # for k in list(range(1, 6)):
    #
    #     perf.write("threshold = {}\n".format(k))
    #
    #     """ find most similar pairs based on cosine similarity """
    #     top_pairs = top_sim(trainedEmb, word2code, tfIdf_clean, k = k)
    #
    #     """ find the best partition and performance for this threshold """
    #     best_index, best_perfm, best_communities = best_split(top_pairs)
    #
    #     perf.write("performance = {}\n".format(best_perfm))
    #     for i,split in enumerate(best_communities):
    #         perf.write("community {}:".format(i) + " ".join(list(split)) + "\n")
    #
    #     """ update the record """
    #     if best_perfm > perfMax:
    #         perfMax = best_perfm
    #         thresholdOPT = k
    #         communityOPT = best_communities

    """ find most similar pairs based on cosine similarity """
    top_pairs = top_sim(trainedEmb, word2code, tfIdf_clean, k=1)

    """ find the best partition and performance for this threshold """
    best_index, best_perfm, best_communities = best_split(top_pairs)
    community_list = []
    for community in best_communities:
        community_list.append(community)

    return trainedEmb, word2code, community_list

def main():

    year_0 = ["2011", "2012", "2013", "2014", "2015", "2016", "2017"]
    month_0 = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    year_1 = ["2011"]
    month_1 = ["01", "02"]
    embed = None
    w2c = None
    communities = {}

    for year in year_0:
        for month in month_0:

            """ examine if file exist """
            filePath = "../../../jiaqima/BlockChain" + year + "-" + month + ".bz2"
            if not os.path.isfile(filePath):
                break

            """ calculate word embeddings for this month, and find the communities """
            print("processing the data in {} {}".format(year, month))
            embed, w2c, community = detect_communities(year, month, embed, w2c)
            communities[int(year+month)] = community

    json.dump(communities, open("results/communities.json", 'w'))

    return

if __name__ == "__main__":
    main()

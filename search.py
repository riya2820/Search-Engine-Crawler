import pickle
import os
import math
import time
from pprint import pprint
from scipy import spatial
from nltk.tokenize import word_tokenize
from indexer import indexer, getTokenPath
from collections import defaultdict
from nltk.stem.porter import *
import traceback

INDEX_ROOT_PATH = "/Users/XXX" # update path
URL_DICT_PATH = "/Users/XXX/index/urls.data" # update path
SECTION_REGULAR_WEIGHT = 1
SECTION_BOLD_WEIGHT = 1.25
SECTION_TITLE_WEIGHT = 1.5
SECTION_HEADING_WEIGHT = 1.75
NUM_TOTAL_DOCS = 55394

def cosineSimilarity(query_tfidf, all_tfidf_in_doc):
    if 0 in query_tfidf:
        return 0
    return (1 - spatial.distance.cosine(query_tfidf, all_tfidf_in_doc))

def retriever(tokens_list):
    # for each token in list, get postings list
    all_tokens_postings = []
    query_tokens_freq = defaultdict(int)
    for token in tokens_list:
        query_tokens_freq[token] += 1

    # calculate query tfidf
    query_tfidf = []
    for token in tokens_list:
        token_file_path = INDEX_ROOT_PATH + "/" + getTokenPath(token)
        if not os.path.exists(token_file_path):
            return []
        with open(token_file_path, 'rb') as path:
            token_posting = pickle.load(path)
            all_tokens_postings.append(token_posting)
            query_tfidf.append((1 + math.log10(query_tokens_freq[token])) * math.log10(NUM_TOTAL_DOCS / len(token_posting)))
    
    # take all curent postings and put in list
    # check if docs of all curr postings are the same
    # if so: add doc id in results list, and increment all indexes in index list
    # else: find first min doc id and increment corresponding index in index list
    # clear curr postings   
    results_list = {}
    curr_postings = []
    index_list = [0 for token in tokens_list]
    end_not_reached = True

    # while each token's posting list has not been fully traversed
    while end_not_reached:
        # for all tokens postings lists, add curr posting pointed to by index_list[i]
        for i in range(len(all_tokens_postings)):
            curr_postings.append(all_tokens_postings[i][index_list[i]])
        # if doc IDs of all curr postings match, then increment all index pointers
        if all(x[0] == curr_postings[0][0] for x in curr_postings):
            for i in range(len(index_list)):
                index_list[i] += 1
                # ensure that new index pointer does not go out of range of token posting list
                if index_list[i] >= len(all_tokens_postings[i]):
                        end_not_reached = False
                        break
            # add doc ID to results list
            posting_for_doc = curr_postings[0]
            all_tfidf_in_doc = []
            for posting in curr_postings:
                all_tfidf_in_doc.append((posting[1] * SECTION_REGULAR_WEIGHT) + (posting[2] * SECTION_TITLE_WEIGHT) + (posting[3] * SECTION_HEADING_WEIGHT) + (posting[4] * SECTION_BOLD_WEIGHT))
            cos_sim_val = cosineSimilarity(query_tfidf, all_tfidf_in_doc)
            results_list[posting_for_doc[0]] = cos_sim_val
        # if doc IDs don't match, then increment index pointer of all postings lists that
        # currently point to a posting with min doc ID
        else:
            min_doc_id = min([x[0] for x in curr_postings])
            for i, posting in enumerate(curr_postings):
                if posting[0] == min_doc_id:
                    index_list[i] += 1
                    if index_list[i] >= len(all_tokens_postings[i]):
                        end_not_reached = False
                        break
        # clear curr posting so it can be rebuilt with new index pointers postings
        curr_postings = []
    return results_list

def main():
    stemmer = PorterStemmer()

    # build index
    indexer()
    url_dict = {}
    with open(URL_DICT_PATH, 'rb') as file:
        url_dict = pickle.load(file)

    # get user input
    while True:
        user_input = input("Enter query: ")
        print("\n")
        start_time = time.time()
        
        # parse and stem tokens and send to retriever
        tokens_list = word_tokenize(user_input)
        for index, token in enumerate(tokens_list):
            tokens_list[index] = stemmer.stem(token).lower()

        # fetch urls of docs and print results
        result_docs = retriever(tokens_list)
        total_time = time.time() - start_time
        print("Search time: " + str(total_time))
        if not result_docs:
            print("No results found")
        else:
            for doc_id, cos_sim_val in sorted(result_docs.items(), key=lambda x:x[1], reverse=True):
                print(url_dict[doc_id] + "\n")
                # print(url_dict[doc_id] + " " + str(cos_sim_val) + "\n")
        print("\n")

if __name__ == "__main__":
    main()

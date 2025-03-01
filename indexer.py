import os
import sys
import json
import math
import pickle
import shutil
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from collections import defaultdict
from bs4 import BeautifulSoup

# randomized seed
HASH_SEED = os.getenv("PYTHONHASHSEED")
if not HASH_SEED:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

INDEX_THRESHOLD = 50
INDEX_ROOT_PATH = "/Users/XXX" # update path
DATA_ROOT_PATH = "/Users/XXX" # update path
URL_DICT_PATH = "/Users/XXXX/index/urls.data" # update path
SECTION_DOC_ID = 0
SECTION_ALL = 1
SECTION_TITLE = 2
SECTION_HEADING = 3
SECTION_BOLD = 4
SECTIONS_TOTAL = 5

# return list of relative paths to all files within given directory
def getListOfFiles(root_dir, curr_dir):
    listOfFile = os.listdir(root_dir + "/" + curr_dir)
    allFiles = list()

    for entry in listOfFile:
        fullPath = os.path.join(root_dir + "/" + curr_dir, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(root_dir, curr_dir + "/" + entry)
        else:
            allFiles.append(curr_dir + "/" + entry)
    return allFiles

def getCleanText(html):
    text_list = [None] * SECTIONS_TOTAL
    soup = BeautifulSoup(html, 'lxml')

    if soup.title is not None:
        text_list[SECTION_TITLE] = soup.title.string

    bold_tags = soup.find_all('b')
    bold_text = []
    for tag in bold_tags:
        if tag.string:
            bold_text.append(tag.string)
    if len(bold_text) > 0:
        text_list[SECTION_BOLD] = " ".join(bold_text)

    heading_tags = soup.find_all(['h1', 'h2', 'h3'])
    heading_text = []
    for tag in heading_tags:
        if tag.string:
            heading_text.append(tag.string)
    if len(heading_text) > 0:
        text_list[SECTION_HEADING] = " ".join(heading_text)

    text_list[SECTION_ALL] = soup.get_text(strip=True)
    return text_list

# get path of directories and filename for token using hash value
def getTokenPath(token):
    # get hash value of token
    # convert to hex string and truncate the first 2 characters (which are 0x)
    # pad with 0s to make 16 character string
    hex_string = hex(hash(token))[2:].zfill(16)
    # creates list using pairs of characters from hex string
    hex_list = list (hex_string[0+i:2+i] for i in range(0, len(hex_string), 2))
    # join list values with / to create relative filepath and filename
    return "/".join(hex_list) + ".data"

def writeToFile(filepath, postings_list):
    # create full filepath and directories for token 
    f_dir = os.path.dirname(filepath)
    if f_dir:
        if not os.path.exists(f_dir):
            os.makedirs(f_dir)
    # write to file in binary to optimize disk usage
    with open(filepath, "wb") as file:
        # serialize postings list using pickle for given token and write to file
        pickle.dump(postings_list, file)

# take current contents of inverted index in memory and write to disk
def offloadIndex(inverted_index, partition_num):
    for token in inverted_index:
        tokenPath = INDEX_ROOT_PATH + "/part" + str(partition_num) + "/" + getTokenPath(token)
        writeToFile(tokenPath, inverted_index[token])

# takes postings lists from source partition and merges with data in destination partition
# using corresponding filepaths for each token
# this function will really only be called with partition 0 as the destination
def mergeIndex(source_num, dest_num):
    token_file_path_list = getListOfFiles(INDEX_ROOT_PATH + "/part" + str(source_num), "")
    for token_file_path in token_file_path_list:
        source_file_path = INDEX_ROOT_PATH + "/part" + str(source_num) + token_file_path
        dest_file_path = INDEX_ROOT_PATH + "/part" + str(dest_num) + token_file_path
        if os.path.isdir(dest_file_path):
            continue
        if os.path.exists(dest_file_path):
            with open(dest_file_path, 'rb') as dest:
                dest_postings_list = pickle.load(dest)
        else:
            f_dir = os.path.dirname(dest_file_path)
            if f_dir:
                if not os.path.exists(f_dir):
                    os.makedirs(f_dir)
            dest_postings_list = []
        with open(source_file_path, 'rb') as source:
            source_postings_list = pickle.load(source)

        dest_postings_list.extend(source_postings_list)
        with open(dest_file_path, 'w+b') as dest:
            pickle.dump(dest_postings_list, dest)

# merge all index partitions and delete source partition once it's empty
def mergeIndexes(num_partitions):
    for part_num in range(1, num_partitions + 1):
        mergeIndex(part_num, 0)
        shutil.rmtree(INDEX_ROOT_PATH + "/part" + str(part_num))

def calculateTfidf(num_docs):
    partition_path_list = getListOfFiles(INDEX_ROOT_PATH + "/part0", "")
    for token_file_path in partition_path_list:
        source_file_path = INDEX_ROOT_PATH + "/part0" + token_file_path
        with open(source_file_path, 'r+b') as source:
            postings_list = pickle.load(source)
            for i, posting in enumerate(postings_list):
                posting[SECTION_ALL] = math.log10(num_docs / len(postings_list)) * posting[SECTION_ALL]
                posting[SECTION_BOLD] = math.log10(num_docs / len(postings_list)) * posting[SECTION_BOLD]
                posting[SECTION_TITLE] = math.log10(num_docs / len(postings_list)) * posting[SECTION_TITLE]
                posting[SECTION_HEADING] = math.log10(num_docs / len(postings_list)) * posting[SECTION_HEADING]
                postings_list[i] = posting
            pickle.dump(postings_list, source)

def indexer():
    path_list = getListOfFiles(DATA_ROOT_PATH, "")
    num_total_postings = 0
    num_partitions = 0
    inverted_index = defaultdict(list)
    doc_id = 0
    stemmer = PorterStemmer()
    url_dict = {}    

    for source_file_path in path_list:
        freq_dict_list = [None] * SECTIONS_TOTAL
        source_file_path = DATA_ROOT_PATH + "/" + source_file_path
        if os.path.isfile(source_file_path):
            with open(source_file_path, encoding='utf-8') as f:
                data = json.load(f)
                url_dict[doc_id] = data['url']
                sections_text = getCleanText(data['content'])
                for s in range(1,SECTIONS_TOTAL):
                    section_freq_dict = defaultdict(int)
                    if sections_text[s] is None:
                        continue
                    for token in word_tokenize(sections_text[s]):
                        # stem each token using porter stemming method
                        token = stemmer.stem(token).lower()
                        section_freq_dict[token] += 1
                    freq_dict_list[s] = section_freq_dict
        
        for token, all_freq in freq_dict_list[SECTION_ALL].items():
            posting = [None] * SECTIONS_TOTAL
            posting[SECTION_DOC_ID] = doc_id
            posting[SECTION_ALL] = all_freq
            if freq_dict_list[SECTION_BOLD] is not None:
                posting[SECTION_BOLD] = freq_dict_list[SECTION_BOLD].get(token, 0)
            else:
                posting[SECTION_BOLD] = 0
            if freq_dict_list[SECTION_TITLE] is not None:
                posting[SECTION_TITLE] = freq_dict_list[SECTION_TITLE].get(token, 0)
            else:
                posting[SECTION_TITLE] = 0
            if freq_dict_list[SECTION_HEADING] is not None:
                posting[SECTION_HEADING] = freq_dict_list[SECTION_HEADING].get(token, 0)
            else:
                posting[SECTION_HEADING] = 0
            inverted_index[token].append(posting)
            num_total_postings += 1
        doc_id += 1
        if doc_id == 5:
            break

        # if index contains certain number of postings, write it to disk
        # create new index partition for each offload operation
        if num_total_postings >= INDEX_THRESHOLD:
            offloadIndex(inverted_index, num_partitions)
            inverted_index = defaultdict(list)
            num_partitions += 1

    offloadIndex(inverted_index, num_partitions)
    mergeIndexes(num_partitions)
    calculateTfidf(doc_id + 1)
    file = open(URL_DICT_PATH, "wb")
    pickle.dump(url_dict, file)
    file.close()

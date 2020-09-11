import os
import re
import subprocess
import pandas as pd
import numpy as np
import scipy
import jieba
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
from collections import defaultdict

import pylcs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel



# load pretrained models
# model_en for english
# model_zh for chinese

#model_zh = SentenceTransformer('distiluse-base-multilingual-cased')



######### Cosine Similarity #########
def cal_cosine(text1, text2, language_type):
    if language_type == 'chinese':
        text1_list = jieba.lcut(text1)
        text2_list = jieba.lcut(text2)
    elif language_type == 'english':
        text1_list = word_tokenize(text1)
        text2_list = word_tokenize(text2)
    else:
        print("ERROR!, there is no suitable language type")
    
    
    l1 = []; l2 = []
    
    text1_set = set(text1_list)
    text2_set = set(text2_list) # also add stop words into the distance
    
    rvector = text1_set.union(text2_set)
    for w in rvector:
        if w in text1_set: l1.append(1)
        else: l1.append(0)
        if w in text2_set: l2.append(1)
        else: l2.append(0)
    c = 0
  
    # cosine formula  
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    cosine = c / float((sum(l1)*sum(l2))**0.5) 

    return cosine, text1_set, text2_set

######### Levenshtein Similarity #########
def cal_levenshtein(str1, str2):
    """
    dynamic programming version
    :param str1
    :param str2
    :return:
    """
    matrix = [[ i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if(str1[i-1] == str2[j-1]):
                d = 0
            else:
                d = 1
            
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)
    distance = matrix[len(str1)][len(str2)]
    normalized_dis = distance / max(len(str1), len(str2))
    leven_similarity = 1 - normalized_dis
    return leven_similarity

########## Constituency/Dependency Parse Distance ###############

def dis_dependency(dep_dict1, dep_dict2):
    dis = 0
    keys = dep_dict1.keys() |  dep_dict2.keys()
    
    for k in keys:
        dis += abs(dep_dict1[k] - dep_dict2[k])
    return dis


def constituent_dict(Constituency):
    components = re.findall(r'\((\w*?)\s', Constituency)
    cons_dict = defaultdict(int)
    for comp in components:
        cons_dict[comp] += 1
    return cons_dict

def constituent_leaf_dict(Constituency):
#     components = re.findall(r'\((\w*?)\s', Constituency)
    components = re.findall(r'\((\w*?)\s\w*\)', Constituency)
    cons_dict = defaultdict(int)
    for comp in components:
        cons_dict[comp] += 1
    return cons_dict

def dependency_dict(dependency):
    dep_dict = defaultdict(int)
    for dep in dependency:
        dep_dict[dep[0]] += 1
    return dep_dict

def cal_constituent_distance(nlp_model, str1, str2):
#     with StanfordCoreNLP(MODEL_DIR,lang=lan) as nlp:
    Constituency1 = nlp_model.parse(str1)
    Constituency2 = nlp_model.parse(str2)
#     dict1 = constituent_dict(Constituency1)
#     dict2 = constituent_dict(Constituency2)
    dict1 = constituent_leaf_dict(Constituency1)
    dict2 = constituent_leaf_dict(Constituency2)
    cons_dis = dis_dependency(dict1, dict2)
    # print("constituency distance = {}".format(cons_dis))
    return cons_dis

def cal_dependency_distance(nlp_model, str1, str2):
#     with StanfordCoreNLP(MODEL_DIR,lang=lan) as nlp:
    Dependency1 = nlp_model.dependency_parse(str1)
    Dependency2 = nlp_model.dependency_parse(str2)
    dep_dict1 = dependency_dict(Dependency1)
    dep_dict2 = dependency_dict(Dependency2)

    dep_dis = dis_dependency(dep_dict1, dep_dict2)
    # print("dependency distance = {}".format(dep_dis))
    return dep_dis

############ LCS Similarity ###############
import pylcs

def cal_lcs(text1, text2):
    return pylcs.lcs(text1, text2)/max(len(text1), len(text2))
    
############ TF-IDF Similarity ##############
# In this similarity calculation, we use the initial seed as the corpus to calculate words' weight

def initialize_vectorizer():
    # get initial seeds
    path = "../data/valid_re_sampling/all_sim_2.csv"
    df = pd.read_csv(path)
    initial_seed = df.to_dict(orient="index")
    initial_seed = initial_seed.values()
    initial_seed = list(initial_seed)
    corpus = []
    for i in initial_seed:
        del i['Unnamed: 0']
        corpus.append(i["NL1"])
        corpus.append(i["NL1'noUnk"])
    # get the corpus, corpus will be the NL1 and NL1' from initial seeds
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    return vectorizer

def cal_tfidf(text1, text2, vectorizer):
    # calculate the tf-idf value of each sentence, the result is normalized
    tf_result = vectorizer.transform([text1, text2]).toarray()
    # calculate the cosine similarity
    cosine_similarity = linear_kernel(tf_result[0].reshape(1,-1), tf_result[1].reshape(1,-1)).flatten()
    return cosine_similarity[0]

############### BLEU Similarity ###############
def cal_bleu(text1, text2):
    bleu_score1 = nltk.translate.bleu_score.sentence_bleu([text1], text2)
    bleu_score2 = nltk.translate.bleu_score.sentence_bleu([text2], text1)
    return max(bleu_score1, bleu_score2)


############## SBERT Similarity ###############

def cal_semantic_sim(model_en, NL1, NL2):
    embedding = model_en.encode([NL1, NL2])
    return abs(1.0 - scipy.spatial.distance.cdist([embedding[0]], [embedding[1]], "cosine")[0])


############### Our Similarity ###############

def unprocess_regex(regex):
    regex = regex.replace("{ ,", "{ 0 ,")
    regex = regex.replace("<VOW>", " ".join('AEIOUaeiou'))
    regex = regex.replace("<NUM>", " ".join('0-9'))
    regex = regex.replace("<LET>", " ".join('A-Za-z'))
    regex = regex.replace("<CAP>", " ".join('A-Z'))
    regex = regex.replace("<LOW>", " ".join('a-z'))

    regex = regex.replace("<M0>", " ".join('dog'))
    regex = regex.replace("<M1>", " ".join('truck'))
    regex = regex.replace("<M2>", " ".join('ring'))
    regex = regex.replace("<M3>", " ".join('lake'))

    regex = regex.replace(" ", "")
    return regex

def is_valid_re(regex):
    try:
        re.compile(regex)
        if len(re.findall(r'\{[\D,]*\}', regex)) != 0 or len(re.findall(r'[&\|]$', regex)) != 0:
            return False
        return  True
    except re.error:
        return False

# DFA Similarity
def cal_dfa(regex1, regex2):
    #print("working on {}".format(regex1))
    regex1 = unprocess_regex(regex1)
    regex2 = unprocess_regex(regex2)
    if not is_valid_re(regex1) or not is_valid_re(regex2):
        #print('invalid re: ', regex1, regex2)
        return False
    
    try:
        command = ['java', '-jar', 'DFASimilarity.jar', '-r1', '{}'.format(regex1), '-r2', '{}'.format(regex2)]
        out = subprocess.check_output(command, timeout=40)
        
        out = out.decode()
        out = out.split('\n')[-2]
        
        if out.startswith("time"):
            out = out[:-2]
            if "E-" in out:
                dfa_sim = float(out.split('E-')[0]) * (10 ** -int(out.split('E-')[1]))
            else:
                dfa_sim = float(out.split('\n')[1])
            #print("time out, force output: ", dfa_sim)
        elif "E-" in out:
            dfa_sim = float(out.split('E-')[0]) * (10 ** -int(out.split('E-')[1]))
        else:
            dfa_sim = float(out[:-2])

#         print("dfa sim = ", dfa_sim)
        return dfa_sim
    except Exception as e:
        #print("Exception: ", e)
        #print(' '.join(command))
        return -1.0

# REGEX Similarity
def cal_regex(str1, str2):
    """
    dynamic programming version
    :param str1
    :param str2
    :return:
    """
    
    pattern = re.compile(r"<M0>|<M1>|<M2>|<M3>|\[<VOW>\]|\[<LOW>\]|\[<NUM>\]|\[<CAP>\]|\[<LET>\]|\{.{0,3}\}|\.|\*|\&|\||\(|\)|\~|\[|\]|\+|\\b")

    str1 = str1.replace(" ", "")
    str2 = str2.replace(" ", "")
    str1 = re.findall(pattern, str1)
    str2 = re.findall(pattern, str2)
    
    matrix = [[ i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if(str1[i-1] == str2[j-1]):
                d = 0
            else:
                d = 1
            
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)
    distance = matrix[len(str1)][len(str2)]
    normalized_dis = distance / max(len(str1), len(str2))
    leven_similarity = 1 - normalized_dis
    return leven_similarity


# Hybrid Similarity
def cal_hyb(str1, str2):
    regex = cal_regex(str1, str2)
    dfa = cal_dfa(str1, str2)
    if dfa == False:
        dfa = 0
    elif dfa == -1.0:
        dfa=0
    return 0.5 * regex + 0.5*dfa

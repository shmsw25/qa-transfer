import argparse
import json, io
import xmltodict
import numpy as np
from collections import Counter
import nltk
import os
import six
from tqdm import tqdm
import re

from IPython import embed

from squad.utils import get_word_span, get_word_idx, process_tokens

def main():
    args = get_args()
    prepro(args)

def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    prepro_each(args, 'train')
    prepro_each(args, 'dev')
    prepro_each(args, 'test')

def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "semeval")
    target_dir = "data/semeval"
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    return parser.parse_args()


def replace_sent(sent):
    return sent.replace("'''", '"').replace("```", '"').replace("''", '"').replace("``", '"') 

def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens

def save(target_dir, data, shared, data_type):
    data_path = os.path.join(target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))

def get_word2vec(glove_dir, word_counter):
    glove_corpus = '6B'
    glove_vec_size= 100
    
    glove_path = os.path.join(glove_dir, "glove.{}.{}d.txt".format(glove_corpus, glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict

def get_exist_words():
    glove_corpus = '6B'
    glove_vec_size= 100

    glove_path = os.path.join("/home/sewon/data/glove", "glove.{}.{}d.txt".format(glove_corpus, glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[glove_corpus]
    words = []
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            words.append(word)
    return set(words)

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

def prepro_each(args, data_type):

    data_list = []
    sub_dir = 'SemEval2016_task3_test/English' if data_type=='test' \
                        else 'v3.2/%s'%(data_type)
    fileName = 'SemEval2016-Task3-CQA-QL-%s-subtaskA.xml'
    if data_type=='train':
        with open(os.path.join(args.source_dir, sub_dir, fileName%('train-part1'))) as f:
            data_list += xmltodict.parse(f.read())['xml']['Thread']
        with open(os.path.join(args.source_dir, sub_dir, fileName%('train-part2'))) as f:
            data_list += xmltodict.parse(f.read())['xml']['Thread']
    else:
        with open(os.path.join(args.source_dir, sub_dir, fileName%(data_type))) as f:
            data_list += xmltodict.parse(f.read())['xml']['Thread']
    questions, comments, answers, question_ids, answer_ids = [], [], [], [], []
    
    text2answer = {'Good':0, 'PotentiallyUseful':1, 'Bad':2}
    for data in data_list:
        q = str(data['RelQuestion']['RelQBody'])
        qid = data['RelQuestion']['@RELQ_ID']
        cs = data['RelComment']
        comment = [c['RelCText'] for c in cs]
        answer = [text2answer[c['@RELC_RELEVANCE2RELQ']] for c in cs]
        cids = [c['@RELC_ID'] for c in cs]
        questions.append(q)
        comments.append(comment)
        answers.append(answer)
        question_ids.append(qid)
        answer_ids.append(cids)

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    char_counter, lower_word_counter = Counter(), Counter()

    print ("start for preprocessing for %s" % (data_type))

    q_len, x_len = [], []
    multi_sent = 0 
 
    for ai, (question, comment, answer, question_id, answer_id) in \
            tqdm(enumerate(zip(questions, comments, answers, question_ids, answer_ids))):
        qi = word_tokenize(question)
        q_len.append(len(qi))
        cqi = [list(qij) for qij in qi]
        for qij in qi:
            lower_word_counter[qij.lower()] += 1
            for qijk in qij:
                 char_counter[qijk] += 1
        for pi, (story, yi, a_id) in enumerate(zip(comment, answer, answer_id)):
            rxi = [ai, pi]
            xi = [word_tokenize(story)]
            x_len.append(len(xi[0]))
            cxi = [list(xij) for xij in xi]
            for xij in xi:
                for xijk in xij:
                    lower_word_counter[xijk.lower()] += 1
                    for xijkl in xijk:
                        char_counter[xijkl] += 1
            json.dump({'x': xi, 'cx': cxi, 'p': story},
                  open(os.path.join(args.target_dir,
                        'shared_%s_%s_%s.json' \
                        % (data_type, str(ai).zfill(3), str(pi).zfill(3))), 'w'))
            def put():
                q.append(qi)
                cq.append(cqi)
                y.append(yi)
                rx.append(rxi)
                rcx.append(rxi)
                ids.append((question_id, a_id))
                idxs.append(len(idxs))
            put()
            if data_type=='train' and yi==0:
                for t in range(3): put()
                if np.random.randint(10)>0: put()
            elif data_type=='train' and yi==2:
                for t in range(3): put()
    lower_word2vec_dict = get_word2vec(args.glove_dir, lower_word_counter)
    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx,
            'idxs': idxs, 'ids': ids, '*p': rx}
    shared = {'char_counter': char_counter,
                'lower_word_counter': lower_word_counter,
                'lower_word2vec': lower_word2vec_dict}
    print("saving ...")
    save(args.target_dir, data, shared, data_type)

if __name__ == '__main__':
    main()




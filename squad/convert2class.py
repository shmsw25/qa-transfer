import argparse
import os
import json

from IPython import embed
import numpy as np
import nltk
from tqdm import tqdm

from squad.utils import process_tokens

def main():
    args = get_args()
    prepro(args, 'train')
    prepro(args, 'dev')

def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "squad")
    target_dir = os.path.join(home, "data", "squad-class")
    parser.add_argument('-s', '--source_dir', default=source_dir)
    parser.add_argument('-t', '--target_dir', default=target_dir)
    return parser.parse_args()

word_tokenize = lambda tokens: [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
sent_tokenize = lambda para: [para]

def _tokenize(c):
    c = c.replace("''", '" ')
    c = c.replace("``", '" ')
    cl = list(map(word_tokenize, sent_tokenize(c)))
    cl = [process_tokens(tokens) for tokens in cl]  # process tokens
    return cl[0]

def getInd(inds):
    ind_set = set(inds)
    if not (len(ind_set)==1 or len(ind_set)==2):
        return None
    if len(ind_set)==1: return inds[0]
    maxCnt = 0
    for i in ind_set:
        if inds.count(i)>maxCnt:
            maxCnt = inds.count(i)
            maxVal = i
    return maxVal

def prepro(args, data_type):
    with open(os.path.join(args.source_dir, '%s-v1.1.json' % data_type), 'r') as f:
        data = json.load(f)['data']
    _data_new = []
    correct_cases = 0
    wrong_cases = 0

    for data_ind, d in enumerate(tqdm(data)):
        paras = d['paragraphs']
        paras_new = []
        for para in paras:
            # paras : list, para : dic of (context, qas)
            # qas: list, qa : dic of (answers(list), id, question)
            c = para['context']
            """
            nonEnd = ["citation needed", ":33", ":79 An observer noted:", ":134"]
            quote = [".", "!", "?"]
            for s in nonEnd:
                if c.endswith(s):
                    c = c[:c.find("[citation needed]")]
            l = [0] + [i+1 for i in range(len(c)) \
                    if c[i] == '.' and (i+1==len(c) or c[i+1]==')' or c[i+1]==' ' or c[i+1]=='"' or c[i+1]=="'")]
            l1 = l[:-1]
            l2 = l[1:]
            context_list = [c[s:e] for (s, e) in zip(l1, l2)]
            """
            indexs = []
            i = 0
            context_list = nltk.sent_tokenize(c)
            for context in context_list:
                i = c[i:].find(context)+i
                assert i>=0
                assert c[i:i+len(context)] == context
                indexs.append((i, i+len(context)))
                i += len(context)
                
            qas_list = []
            for i in range(len(context_list)):
                qas_list.append([])
            for qa in para['qas']:
                answers = qa['answers']
                question = qa['question']
                _id = qa['id']
                inds = []
                for answer in answers:
                    start = answer['answer_start']
                    text = answer['text']
                    assert c[start:start+len(text)] == text
                    ind_list = [i for (i, (s, e)) in enumerate(indexs) if (s<=start and start+len(text)<=e)]
                    if len(ind_list)==0:
                        wrong_cases += 1
                        continue
                    elif len(ind_list)==1:
                        correct_cases += 1
                    else:
                        embed()
                    inds.append(ind_list[0])
                ind = getInd(inds)
                if ind is None: continue
                for i in range(len(context_list)):
                    dic = {'id' : _id+str(i), 'question' : question, 'answer' : (i==ind)}
                    qas_list[i].append(dic)
            for context, qas in zip(context_list, qas_list):
                dic = {'context':context, 'qas':qas}
                paras_new.append(dic)
        _data_new.append({'paragraphs' : paras_new})

    print ("Ignore %d wrong cases (among %d) due to the incorrect sentence tokenization." % (wrong_cases, correct_cases+wrong_cases))

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    with open(os.path.join(args.target_dir, '%s-v1.1.json' % data_type), 'w') as f:
        json.dump({'data':_data_new}, f)

if __name__ == '__main__':
    main()




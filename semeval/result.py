import os
import argparse
import json, gzip, pickle
from IPython import embed
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    data_dir = os.path.join("data", "semeval")
    eval_dir = os.path.join("out/semeval", "basic-class")
    store_dir = os.path.join("semeval", "store")
    parser.add_argument('-d', "--data_dir", default=data_dir)
    parser.add_argument('-e', "--eval_dir", default=eval_dir)
    parser.add_argument('-s', "--store_dir", default=store_dir)
    parser.add_argument("--run_ids")
    parser.add_argument("--eval_name", default='test')
    parser.add_argument("--eval_period", type=int, default=200)
    parser.add_argument("--start_step", type=int)
    parser.add_argument("--end_step", type=int)
    parser.add_argument("--steps", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ensemble", action='store_true')
    return parser.parse_args()

def main():
    args = get_args()

    if args.ensemble:
        print ("Ensemble not implemented yet")
        return

    data = load_data(args)
    if not os.path.exists(args.store_dir):
        os.makedirs(args.store_dir)

    for i, run_id in enumerate(args.run_ids.split(',')):
        for step in range(args.start_step, args.end_step, args.eval_period):
            evaluate(args, data, run_id, step, dump_gold=(i==0 and step==args.start_step))

def load_data(args):
    data_path = os.path.join(args.data_dir, "data_%s.json" % args.eval_name)
    return json.load(open(data_path, 'r'))

def evaluate(args, data, run_id, step, ensemble=False, ensemble_list=[], dump_gold=False):

    def get(run_id, number):
        eval_path = os.path.join(args.eval_dir, run_id.zfill(2), 'eval', '%s-%s.pklz' % (args.eval_name, str(step).zfill(6)))
        result = pickle.load(gzip.open(eval_path, 'rb'))
        ys = result['y']
        yps = [float(yp[0][1]) for yp in result['yp']]
        idxs = result['idxs']
        return ys, yps, idxs

    if ensemble:
        id2number= {3:21, 4:19, 5:21, 0:4, 30:4, 31:6, 32:5, 8:21, 11:21, 15:19, \
                        16:21, 18:22, 14:22, 19:22, 20:19, 21:19, 24:19}
        for i, run_id in enumerate(ensemble_list):
            number = id2number.get(run_id, 20)
            ys_, yps_, idxs_ = get(run_id, number)
            if i==0:
                yps, ys, idxs = yps_, ys_, idxs_
            else:
                for j, yp_ in enumerate(yps_):
                    yps[j] += yp_
                assert ys==ys_ and idxs==idxs_
        for i, yp in enumerate(yps):
            yps[i] = yp/len(ensemble_list)
    else:
        ys, yps, idxs = get(run_id, step)

    if ensemble:
        store_path = os.path.join(args.store_dir, 'ensemble')
    else:
        store_path = os.path.join(args.store_dir, '%s-%s-%s' % (args.eval_name, str(run_id).zfill(2), str(step).zfill(6)))
    if dump_gold:
        gold_store_path = os.path.join(args.store_dir, '%s-gold' % args.eval_name)
    else:
        gold_store_path = None

    data_ys = data['y']
    data_rxis = data['*x']
    data_ids = data['ids']

    total_list = []
    curr_ai, curr_qid = None, None
    for i, (idx, _y, _yp, data_y, data_rxi, data_id) in \
                enumerate(zip(idxs, ys, yps, data_ys, data_rxis, data_ids)):
        assert _y==data_y
        if curr_ai == data_rxi[0]:
            assert data_rxi[1] == len(this_list)
            assert curr_qid == data_id[0]
            this_list.append((_y, _yp, data_id[1]))
        else:
            if i>0: total_list.append((curr_qid, this_list))
            this_list = [(_y, _yp, data_id[1])]
            curr_ai = data_rxi[0]
            curr_qid = data_id[0]
            assert data_rxi[1]==0
        if i+1==len(idxs):
            total_list.append((curr_qid, this_list))
    get_score(args, data, total_list, store_path, gold_store_path)

def get_score(args, data, total_list, store_path, gold_store_path=None):
    predf = open(store_path, 'w')
    if gold_store_path is not None:
        goldf = open(gold_store_path, 'w')

    data_ys = data['y']
    data_rxis = data['*x']
    data_ids = data['ids']

    correct, wrong = 0, 0 # for p@1
    pred = [] # for MAP
    inv_rank = [] # for MRR
    accs = [] # Accuracy
    ps, rs, f1s = [], [], []
    for ii, (qid, this_list) in enumerate(total_list):
        # this_list is a list of tuples (y, yp)
        ys = [l[0] for l in this_list] # true answers (which is 0/1/2)
        yps = [l[1] for l in this_list] # prob of true answer (prob of Good)
        aids = [l[2] for l in this_list]
        def bool2lower(b):
            return 'true' if b else 'false'
        for jj, (aid, yp, y) in enumerate(zip(aids, yps, ys)):
            line = "%s\t%s\t%s\t%s\t%s\n"%(qid,aid,0,yp,bool2lower(yp>=args.threshold))
            predf.write(line)
            if gold_store_path is not None:
                line = "%s\t%s\t%s\t%s\t%s\n"%(qid,aid,0,0,bool2lower(y==0))
                goldf.write(line)

    predf.close()
    if gold_store_path is not None:
        goldf.close()

if __name__ == '__main__':
    main()

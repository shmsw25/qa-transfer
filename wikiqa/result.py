import argparse
import os
import json, gzip, pickle
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    data_dir = os.path.join(home, "data", "WikiQACorpus")
    eval_dir = "out/wikiqa/basic-class"
    parser.add_argument('-d', "--data_dir", default=data_dir)
    parser.add_argument('-e', "--eval_dir", default=eval_dir)
    parser.add_argument("--run_ids")
    parser.add_argument("--eval_name", default='test')
    parser.add_argument("--eval_period", type=int, default=200)
    parser.add_argument("--start_step", type=int)
    parser.add_argument("--end_step", type=int)
    parser.add_argument("--steps", default="")
    parser.add_argument("--ensemble", action='store_true')
    return parser.parse_args()

def main():
    args = get_args()
    data = load(args)

    if args.steps:
        run_ids = args.run_ids.split(',')
        steps = args.steps.split(',')
        if not len(run_ids)==len(steps):
            print ("Number of run_ids and global_steps are different.")
            return
        if args.ensemble:
            evaluate(args, run_ids, data, steps, print_result=True)
        else:
            for (run_id, step) in zip(run_ids, steps):
                evaluate(args, [run_id], data, [step], print_result=True)
    else:
        for run_id in args.run_ids.split(','):
            best_eval, best_global_step = (0, 0, 0), -1
            print ("Evaluate run_id = %s..." % run_id)
            for global_step in range(args.start_step, args.end_step + args.eval_period, args.eval_period):
                curr_eval = evaluate(args, [run_id], data, [global_step])
                if curr_eval[0] > best_eval[0]:
                    best_eval, best_global_step = curr_eval, global_step
            print ("Best MAP: %.2f\tMRR: %.2f\tP@1: %.2f in global step %d" % (best_eval[0], best_eval[1], best_eval[2], best_global_step))

def evaluate(args, run_ids, data, global_steps, print_result=False):
    eval_lists = []
    for run_id, global_step in zip(run_ids, global_steps):
        eval_path = os.path.join(args.eval_dir, run_id.zfill(2), 'eval', '%s-0%s.pklz' % (args.eval_name, str(global_step).zfill(5)))
        eval_lists.append(load_eval(data, eval_path))
    eval_list = []
    if len(run_ids) > 1:
        for i in range(len(eval_lists[0])):
            each_eval_list = []
            for j in range(len(eval_lists[0][i])):
                each_eval_list.append((eval_lists[0][i][j][0], sum([each_runid[i][j][1] for each_runid in eval_lists])))
            eval_list.append(each_eval_list)
    else:
        eval_list = eval_lists[0]
    MAP, MRR, P1 = get_score(data, eval_list)
    if print_result:
        print ("global step %s\tMAP: %.3f\tMRR: %.3f\tP@1: %.3f" % (global_step, MAP, MRR, P1))
    return MAP, MRR, P1

def load(args):
    data_dic = {}
    with open(os.path.join(args.data_dir, '%s-class.json' % args.eval_name), 'r') as f:
        data = json.load(f)
        data = data['data']
        for _data in data:
            dic = _data['paragraphs'][0]
            context = dic['context']
            qas = dic['qas'][0]
            answer = qas['answer']
            _id = qas['id']
            question = qas['question']
            data_dic[_id] = {'c' : context, 'q' : question, 'a' : answer}
    return data_dic

def load_eval(data_dic, eval_path):
    result = pickle.load(gzip.open(eval_path, 'rb'))
    ys = result['y']
    yps = result['yp']
    idxs = result['idxs']

    total_list = []
    curr_q = None
    for i, (idx, y, yp) in enumerate(zip(idxs, ys, yps)):
        _y = int(y[0])
        _yp = float(yp[0][1])
        q = data_dic[idx]['q']
        if curr_q == q:
            this_list.append((_y, _yp))
        else:
            if i>0: total_list.append(this_list)
            this_list = [(_y, _yp)]
            curr_q = q
        if i+1==len(idxs):
            total_list.append(this_list)
    return total_list

def get_score(data_dic, total_list):
    correct, wrong = 0, 0 # for p@1
    pred = [] # for MAP
    inv_rank = [] # for MRR
    for this_list in total_list:
        # this_list is a list of tuples (y, yp)
        ys = [l[0] for l in this_list] 	# true answers
        yps = [l[1] for l in this_list] # prob of true answer
        if not 1 in ys: continue 	# remove cases of no answer
					# following previous works
        my_preds = [yp for (y, yp) in zip(ys, yps) if y==1]

        yps.sort(reverse=True)

        rank = len(yps)
        for i in my_preds:
            if rank>yps.index(i): rank=yps.index(i)
        rank += 1			# model set groundtruth which rank

        inv_rank.append(1.0/float(rank))# for MRR
        if rank==1: correct+=1		
        else: wrong += 1		# for P@1

        precs = []
        for i, ypi in enumerate(yps):
            if ypi in my_preds:
                prec = (1.0+len(precs))/(i+1.0)
                precs.append(prec)
        if len(precs)==0: pred.append(0.0)
        else: pred.append(np.mean(precs))

    MAP = np.mean(pred)*100
    MRR = np.mean(inv_rank)*100
    P1 = float(correct)/float(correct+wrong)*100
    return (MAP, MRR, P1)


if __name__ == '__main__':
    main()


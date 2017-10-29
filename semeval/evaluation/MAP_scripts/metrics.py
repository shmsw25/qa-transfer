from __future__ import division


def mrr(out, th):
  """Computes MRR.

  Args:
    out: dict where each key maps to a ranked list of candidates. Each values
    is "true" or "false" indicating if the candidate is relevant or not.
  """
  n = len(out)
  MRR = 0.0
  for qid in out:
    candidates = out[qid]
    for i in xrange(min(th, len(candidates))):
      if candidates[i] == "true":
        MRR += 1.0 / (i + 1)
        break
  return MRR / n


def precision(out, th):
  precisions = [0.0]*th
  n = 0
  for qid in out:
    candidates = out[qid]
    if all(x == "false" for x in candidates):
      continue
    for i in xrange(min(th, len(candidates))):
      if candidates[i] == "true":
        precisions[i] += 1.0
        break
    n += 1
  for i in xrange(1, th):
    precisions[i] += precisions[i-1]  

  return [p*100/n for p in precisions]


def recall_of_1(out, th):
  precisions = [0.0]*th
  for qid in out:
    candidates = out[qid]
    if all(x == "false" for x in candidates):
      continue
    for i in xrange(min(th, len(candidates))):
      if candidates[i] == "true":
        precisions[i] += 1.0
        break
  for i in xrange(1, th):
    precisions[i] += precisions[i-1]  

  return [p*100/len(out) for p in precisions]


def map(out, th):
  num_queries = len(out)
  MAP = 0.0
  for qid in out:
    candidates = out[qid]
    # compute the number of relevant docs
    # get a list of precisions in the range(0,th)
    avg_prec = 0
    precisions = []
    num_correct = 0
    for i in xrange(min(th, len(candidates))):
      if candidates[i] == "true":
        num_correct += 1
        precisions.append(num_correct/(i+1))
    
    if precisions:
      avg_prec = sum(precisions)/len(precisions)
    
    MAP += avg_prec
  return MAP / num_queries

  
def accuracy(out, th):
  """Computes accuracy, i.e. number of correct answers retrieved at rank @X. 

  Accuracy is normalized by the rank and the total number of questions.
  """
  acc = [0.0]*th
  for qid in out:
    candidates = out[qid]
    for i in xrange(min(th, len(candidates))):
      if candidates[i] == "true":
        acc[i] += 1.0
  for i in xrange(1, th):
    acc[i] += acc[i-1]  
  
  return [a*100/(i*len(out)) for i, a in enumerate(acc, 1)]


def accuracy1(out, th):
  """Accuracy normalized by the number of maximum possible answers.

  The number of correct answers at @X normalized by the number of maximum 
  possible answers (perfect re-ranker).
  """
  acc = [0.0]*th
  maxrel = [0.0]*th
  for qid in out:
    relevant = out[qid]
    num_relevant = sum([1.0 for x in relevant if x == "true"])
    # print num_relevant
    for i in xrange(min(th, len(relevant))):
      if relevant[i] == "true":
        acc[i] += 1.0   
    for i in xrange(th):
      maxrel[i] += min(i+1, num_relevant)
  for i in xrange(1, th):
    acc[i] += acc[i-1]  
  return [a/numrel for a, numrel in zip(acc, maxrel)]

def avg_acc1(out, th):
  acc = accuracy1(out, th)
  return sum(acc)/len(acc)

def accuracy2(out, th):
  """Accuracy - the absolute number of correct answers at @X.
  """
  acc = [0.0]*th
  for qid in out:
    relevant = out[qid]
    for i in xrange(min(th, len(relevant))):
      if relevant[i] == "true":
        acc[i] += 1.0
  for i in xrange(1, th):
    acc[i] += acc[i-1]  
  return [a for i, a in enumerate(acc, 1)]

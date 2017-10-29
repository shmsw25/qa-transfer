#! /usr/bin/env python

import sys
import logging
from collections import defaultdict
from operator import itemgetter
from optparse import OptionParser
from res_file_reader import ResFileReader
import metrics

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_res_file(res_fname, format):
	logging.info("Processing file: %s with search engine ranks" % res_fname)
	lineReader = ResFileReader(format)

	ir = defaultdict(list)
	for line_res in open(res_fname):
		qid, aid, relevant, ir_score = lineReader.read_line(line_res)  # process the line from the res file
		ir[qid].append( (relevant, ir_score) )
	
	# Sort based on the search engine score (largest to smallest).
	for qid, resList in ir.iteritems():
		ir[qid] = [rel for rel, score in sorted(resList, key = itemgetter(1), reverse = True)]  
	return ir


def read_res_pred_files(res_fname, pred_fname, format, verbose=True, 
                        reranking_th=0.0, 
                        ignore_noanswer=False):

	lineReader = ResFileReader(format)
	lineReader_pred = ResFileReader(format)

	ir, svm = defaultdict(list), defaultdict(list)
	conf_matrix = {'true' : {'true' : 0, 'false' : 0}, 'false' : {'true' : 0, 'false' : 0}}
	lineNo = 0
	for line_res, line_pred in zip(open(res_fname), open(pred_fname)):
		lineNo = lineNo + 1
		# Process the line from the res file.
		qid, aid, relevant, ir_score = lineReader.read_line(line_res)
		pred_qid, pred_aid, pred_relevant, pred_score = lineReader_pred.read_line(line_pred)
		if (qid != pred_qid) or (aid != pred_aid):
			print 'ERROR: ID mismatch on line ' + str(lineNo) + ':'
			print 'in ' + res_fname + ' we have (' + qid + ',' + aid + '),'
			print 'but in ' + pred_fname + ' we have (' + pred_qid + ',' + pred_aid + ')'
			quit()

		if (relevant != 'true') and (relevant != 'false'):
			print 'ERROR: wrong label on line ' + str(lineNo) + ' in ' + res_fname + ': "' + relevant + '"'
			print 'Allowed values are only "true" and "false"'
			quit()

		if (pred_relevant != 'true') and (pred_relevant != 'false'):
			print 'ERROR: wrong label on line ' + str(lineNo) + ' in ' + pred_fname + ': "' + pred_relevant + '"'
			print 'Allowed values are only "true" and "false"'
			quit()

		ir[qid].append( (relevant, ir_score, aid) )
		svm[qid].append( (relevant, pred_score, aid) )
		conf_matrix[relevant][pred_relevant] = conf_matrix[relevant][pred_relevant] + 1

	if verbose:
		analyze_file = open(pred_fname + ".analysis", "w")
		info_file = open(pred_fname + ".correctpos", "w")

	# Remove questions that contain no correct answer
	if ignore_noanswer:
		for qid in ir.keys():
			candidates = ir[qid]
			if all(relevant == "false" for relevant,_,_ in candidates):
				del ir[qid]
				del svm[qid]

	for qid in ir:
		# Sort by IR score.
		ir_sorted = sorted(ir[qid], key = itemgetter(1), reverse = True)
		
		# Sort by SVM prediction score.
		svm_sorted = svm[qid]
		max_score = max([score for rel, score, aid in svm_sorted])
		if max_score >= reranking_th:
			svm_sorted = sorted(svm_sorted, key = itemgetter(1), reverse = True)

		if verbose:
			before = find_correct_answer_position(ir_sorted)
			after = find_correct_answer_position(svm_sorted)
			impr = analyze_reranking_improvement(before, after)
			analyze_file.write("%s %s\n" % (qid, str(impr)))
			info_file.write("%s %s %s\n" % (qid, str(before), str(after)))

		ir[qid] = [rel for rel, score, aid in ir_sorted]
		svm[qid] = [rel for rel, score, aid in svm_sorted]
	
	if verbose:
		analyze_file.close()
		info_file.close()

	return ir, svm, conf_matrix

def find_correct_answer_position(candidates):
	out = {}
	for i, (rel, score, aid) in enumerate(candidates, 1):
		if rel == "true":
			out[aid] = i
	return out

def analyze_reranking_improvement(before, after):
	out = {}
	for key, rank_before in before.iteritems():
		rank_after = after[key]
		improvement = rank_before - rank_after
		out[key] = improvement
	return out


def eval_reranker(res_fname="svm.test.res", pred_fname="svm.train.pred", 
                  format="trec",
                  th=10, 
                  verbose=False,
                  reranking_th=0.0,
                  ignore_noanswer=False):
	ir, svm, conf_matrix = read_res_pred_files(res_fname, pred_fname, format, verbose, 
		                              reranking_th=reranking_th, 
		                              ignore_noanswer=ignore_noanswer)		
	# Calculate standard P, R, F1, Acc
	acc = 1.0 * (conf_matrix['true']['true'] + conf_matrix['false']['false']) / (conf_matrix['true']['true'] + conf_matrix['false']['false'] + conf_matrix['true']['false'] + conf_matrix['false']['true'])
	p = 0
	if (conf_matrix['true']['true'] + conf_matrix['false']['true']) > 0:
		p = 1.0 * (conf_matrix['true']['true']) / (conf_matrix['true']['true'] + conf_matrix['false']['true'])
	r = 0
	if (conf_matrix['true']['true'] + conf_matrix['true']['false']) > 0:
		r = 1.0 * (conf_matrix['true']['true']) / (conf_matrix['true']['true'] + conf_matrix['true']['false'])
	f1 = 0
	if (p + r) > 0:
		f1 = 2.0 * p * r / (p + r)

	# evaluate IR
	prec_se = metrics.recall_of_1(ir, th)
	acc_se = metrics.accuracy(ir, th)
	acc_se1 = metrics.accuracy1(ir, th)
	acc_se2 = metrics.accuracy2(ir, th)

	# evaluate SVM
	prec_svm = metrics.recall_of_1(svm, th)
	acc_svm = metrics.accuracy(svm, th)
	acc_svm1 = metrics.accuracy1(svm, th)
	acc_svm2 = metrics.accuracy2(svm, th)

	mrr_se = metrics.mrr(ir, th)
	mrr_svm = metrics.mrr(svm, th)
	map_se = metrics.map(ir, th)
	map_svm = metrics.map(svm, th)

	avg_acc1_svm = metrics.avg_acc1(svm, th)
	avg_acc1_ir = metrics.avg_acc1(ir, th)

	#print ""
	#print "*** Official score (MAP for SYS): %5.4f" %(map_svm)
	#print ""
	#print ""
	#print "******************************"
	#print "*** Classification results ***"
	#print "******************************"
	#print ""
	#print "Acc = %5.4f" %(acc)
	#print "P   = %5.4f" %(p)
	#print "R   = %5.4f" %(r)
	#print "F1  = %5.4f" %(f1)
	#print ""
	#print ""
	#print "********************************"
	#print "*** Detailed ranking results ***"
	#print "********************************"
	#print ""
	#print "IR  -- Score for the output of the IR system (baseline)."
	#print "SYS -- Score for the output of the tested system."
	#print ""
	#print "%13s %5s" %("IR", "SYS")
	#print "MAP   : %5.4f %5.4f" %(map_se, map_svm)
	#print "AvgRec: %5.4f %5.4f" %(avg_acc1_ir, avg_acc1_svm)
	#print "MRR   : %6.2f %6.2f" %(mrr_se, mrr_svm)
	print "MAP   : %5.4f\tMRR   : %5.4f\tAvgRec: %5.4f" %(map_svm, mrr_svm, avg_acc1_svm)
	#print "Acc   : %5.4f" %(acc)
	#print "P     : %5.4f" %(p)
	#print "R     : %5.4f" %(r)
	#print "F1    : %5.4f" %(f1)
        """
	print "%16s %6s  %14s %6s  %14s %6s  %12s %4s" % ("IR", "SYS", "IR", "SYS", "IR", "SYS", "IR", "SYS")
	for i, (p_se, p_svm, a_se, a_svm, a_se1, a_svm1, a_se2, a_svm2) in enumerate(zip(prec_se, prec_svm, acc_se, acc_svm, acc_se1, acc_svm1, acc_se2, acc_svm2), 1):
		print "REC-1@%02d: %6.2f %6.2f  ACC@%02d: %6.2f %6.2f  AC1@%02d: %6.2f %6.2f  AC2@%02d: %4.0f %4.0f" %(i, p_se, p_svm, i, a_se, a_svm, i, a_se1, a_svm1, i, a_se2, a_svm2)
	print
	print "REC-1 - percentage of questions with at least 1 correct answer in the top @X positions (useful for tasks where questions have at most one correct answer)"
	print "ACC   - accuracy, i.e., number of correct answers retrieved at rank @X normalized by the rank and the total number of questions"
	print "AC1   - the number of correct answers at @X normalized by the number of maximum possible answers (perfect re-ranker)"
	print "AC2   - the absolute number of correct answers at @X"
	"""

def eval_search_engine(res_fname, format, th=10):
	ir = read_res_file(res_fname, format)		

	# evaluate IR
	rec = metrics.recall_of_1(ir, th)
	acc = metrics.accuracy(ir, th)
	acc1 = metrics.accuracy1(ir, th)
	acc2 = metrics.accuracy2(ir, th)

	mrr = metrics.mrr(ir, th)

	print "%13s" %"IR"
	print "MRRof1: %5.2f" % mrr
	for i, (r, a, a1, a2) in enumerate(zip(rec, acc, acc1, acc2), 1):
		print "REC-1@%02d: %6.2f  ACC@%02d: %6.2f  AC1@%02d: %6.2f  AC2@%02d: %4.0f" %(i, r, i, a, i, a1, i, a2)
	print
	print "REC-1 - percentage of questions with at least 1 correct answer in the top @X positions (useful for tasks were questions have at most one correct answer)"
	print "ACC   - accuracy, i.e. number of correct answers retrieved at rank @X normalized by the rank and the total number of questions"
	print "AC1   - the number of correct answers at @X normalized by the number of maximum possible answers (perfect re-ranker)"
	print "AC2   - the absolute number of correct answers at @X"


def main():
	usage = "usage: %prog [options] arg1 [arg2]"
	desc = """arg1: file with the output of the baseline search engine (ex: svm.test.res) 
	arg2: predictions file from svm (ex: train.predictions)
	if arg2 is ommited only the search engine is evaluated"""

	parser = OptionParser(usage=usage, description=desc)
	parser.add_option("-t", "--threshold", dest="th", default=10, type=int, 
	                  help="supply a value for computing Precision up to a given threshold "
	                  "[default: %default]", metavar="VALUE")
	parser.add_option("-r", "--reranking_threshold", dest="reranking_th", default=None, type=float, 
	                  help="if maximum prediction score for a set of candidates is below this threshold, do not re-rank the candiate list."
	                  "[default: %default]", metavar="VALUE")
	parser.add_option("-f", "--format", dest="format", default="trec", 
	                  help="format of the result file (trec, answerbag): [default: %default]", 
	                  metavar="VALUE")	 	  
	parser.add_option("-v", "--verbose", dest="verbose", default=False, action="store_true",
	                  help="produce verbose output [default: %default]")	 	  
	parser.add_option("--ignore_noanswer", dest="ignore_noanswer", default=False, action="store_true",
	                  help="ignore questions with no correct answer [default: %default]")	 	  
	
	(options, args) = parser.parse_args()

	if len(args) == 1:
		res_fname = args[0]
		eval_search_engine(res_fname, options.format, options.th)
	elif len(args) == 2:
		res_fname = args[0]
		pred_fname = args[1]	
		eval_reranker(res_fname, pred_fname, options.format, options.th, 
		              options.verbose, options.reranking_th, options.ignore_noanswer)
	else:
		parser.print_help()
		sys.exit(1)
	

if __name__ == '__main__':	
	main()


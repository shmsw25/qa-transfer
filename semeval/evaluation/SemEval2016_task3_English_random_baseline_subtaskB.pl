#!/usr/bin/perl
#
#  Author: Preslav Nakov
#  
#  Description: Generates output for a random baseline + IR & gold standard labels
#               for SemEval-2016 Task 3, subtask B
#
#  Last modified: October 21, 2015
#
#

# Example run:
#    perl SemEval2016_task3_English_random_baseline_subtaskB.pl SemEval2016-Task3-CQA-QL-dev.xml
#    python MAP_scripts/ev.py SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.pred



use warnings;
use strict;
use utf8;


################
###   MAIN   ###
################

die "Use $0 <INPUT_FILE>" if (0 != $#ARGV);
my $INPUT_FILE = $ARGV[0];
my $OUTPUT_FILE_RELEVANCY_GOLD = $INPUT_FILE . '.subtaskB.relevancy';
my $OUTPUT_FILE_BASELINE       = $INPUT_FILE . '.subtaskB.pred';

### 1. Open the files and 
open INPUT, $INPUT_FILE or die;
open OUTPUT_GOLD, '>' . $OUTPUT_FILE_RELEVANCY_GOLD or die;
open OUTPUT_BASELINE, '>' . $OUTPUT_FILE_BASELINE or die;
binmode(INPUT, ":utf8");
binmode(OUTPUT_GOLD, ":utf8");
binmode(OUTPUT_BASELINE, ":utf8");

srand 0;
while (<INPUT>) {

	# <RelQuestion RELQ_ID="Q1_R4" RELQ_RANKING_ORDER="4" RELQ_CATEGORY="Advice and Help" RELQ_DATE="2013-05-02 19:43:00" RELQ_USERID="U1" RELQ_USERNAME="ankukuma" RELQ_RELEVANCE2ORGQ="PerfectMatch">
	next if !/<RelQuestion RELQ_ID=\"Q([0-9]+)_R([0-9]+)\" [^<>]+ RELQ_RELEVANCE2ORGQ=\"([^\"]+)\">/;
	my ($qid, $relQID, $rel2OrgQ) = ($1, $2, $3);
	my ($rank, $score) = ($2, 1.0 / $2);

	if ($rel2OrgQ eq 'PerfectMatch') { $rel2OrgQ = 'true'; }
	elsif ($rel2OrgQ eq 'Relevant') { $rel2OrgQ = 'true'; }
	elsif ($rel2OrgQ eq 'Irrelevant') { $rel2OrgQ = 'false'; }
	else { die "Wrong value for RELQ_RELEVANCE2ORGQ: '$rel2OrgQ'"; }

	print OUTPUT_GOLD "Q$qid\tQ$qid\_R$relQID\t$relQID\t$score\t$rel2OrgQ\n";

	$score = rand(2);
	my $label = ($score > 0.5) ? 'true' : 'false';
	print OUTPUT_BASELINE "Q$qid\tQ$qid\_R$relQID\t0\t$score\t$label\n";
}

### 3. Close the files
close INPUT or die;
close OUTPUT_GOLD or die;
close OUTPUT_BASELINE or die;

#!/usr/bin/perl
#
#  Author: Preslav Nakov
#  
#  Description: Generates output for a random baseline + IR & gold standard labels
#               for SemEval-2016 Task 3, subtask C
#
#  Last modified: October 21, 2015
#
#

# Example run:
#    perl SemEval2016_task3_English_random_baseline_subtaskC.pl SemEval2016-Task3-CQA-QL-dev.xml
#    python MAP_scripts/ev.py SemEval2016-Task3-CQA-QL-dev.xml.subtaskC.relevancy SemEval2016-Task3-CQA-QL-dev.xml.subtaskC.pred


use warnings;
use strict;
use utf8;


################
###   MAIN   ###
################

die "Use $0 <INPUT_FILE>" if (0 != $#ARGV);
my $INPUT_FILE = $ARGV[0];
my $OUTPUT_FILE_RELEVANCY_GOLD = $INPUT_FILE . '.subtaskC.relevancy';
my $OUTPUT_FILE_BASELINE       = $INPUT_FILE . '.subtaskC.pred';

### 1. Open the files and 
open INPUT, $INPUT_FILE or die;
open OUTPUT_GOLD, '>' . $OUTPUT_FILE_RELEVANCY_GOLD or die;
open OUTPUT_BASELINE, '>' . $OUTPUT_FILE_BASELINE or die;
binmode(INPUT, ":utf8");
binmode(OUTPUT_GOLD, ":utf8");
binmode(OUTPUT_BASELINE, ":utf8");

srand 0;
while (<INPUT>) {

	#<RelComment RELC_ID="Q104_R1_C1" RELC_DATE="2010-08-27 01:40:05" RELC_USERID="U8" RELC_USERNAME="anonymous" RELC_RELEVANCE2ORGQ="Good" RELC_RELEVANCE2RELQ="Good">
	next if !/<RelComment RELC_ID=\"Q([0-9]+)_R([0-9]+)_C([0-9]+)\" [^<>]+ RELC_RELEVANCE2ORGQ=\"([^\"]+)\"/;
	my ($qid, $relQID, $relCID, $rel2OrgQ) = ($1, $2, $3, $4);
	my $rank  = $relQID*100 + $relCID;
	my $score = 1.0 / $rank;

	if ($rel2OrgQ eq 'Good') { $rel2OrgQ = 'true'; }
	elsif ($rel2OrgQ eq 'Bad') { $rel2OrgQ = 'false'; }
	elsif ($rel2OrgQ eq 'PotentiallyUseful') { $rel2OrgQ = 'false'; }
	else { die "Wrong value for RELC_RELEVANCE2ORGQ: '$rel2OrgQ'"; }

	print OUTPUT_GOLD "Q$qid\tQ$qid\_R$relQID\_C$relCID\t$rank\t$score\t$rel2OrgQ\n";

	$score = rand(2);
	my $label = ($score > 0.5) ? 'true' : 'false';
	print OUTPUT_BASELINE "Q$qid\tQ$qid\_R$relQID\_C$relCID\t0\t$score\t$label\n";
}

### 3. Close the files
close INPUT or die;
close OUTPUT_GOLD or die;
close OUTPUT_BASELINE or die;

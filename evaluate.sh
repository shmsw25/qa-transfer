#! /bin/bash

DATA=$1
RUN_ID=$2
START_STEP=$3
END_STEP=$4

if [ $DATA = "wikiqa" ]
then
	python -m wikiqa.result --run_ids $RUN_ID --start_step $START_STEP --end_step $END_STEP
elif [ $DATA = "semeval" ]
then
	python -m semeval.result --run_ids $RUN_ID --start_step $START_STEP --end_step $END_STEP
	for ((i=$START_STEP; i<$END_STEP; i=((i+200)))); do
		python2.7 semeval/evaluation/MAP_scripts/ev.py semeval/store/test-gold semeval/store/test-$RUN_ID-0$i
	done
else
	echo "WRONG DATA. [wikiqa | semeval]"
fi



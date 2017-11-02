#! /bin/bash

DATA=$1
TASK=$2
RUN_ID=$3
PRETR_FROM=$4
STEP=$5

PRETR_RUN_ID="00"
PRETR_DIR=out/squad/$PRETR_FROM/$PRETR_RUN_ID
	
if [ $DATA = "wikiqa" ]
then
	DATA_DIR="data/wikiqa-class"
	ADD=""
elif [ $DATA = "semeval" ]
then
	DATA_DIR="data/semeval"
	ADD="--load_shared --nocluster --sent_size_th 150 --ques_size_th 100"
else
	echo "WRONG DATA. [wikiqa | semeval]"
	exit
fi

if [ $TASK = "finetune" ]
then
	python -m basic.cli --data_dir $DATA_DIR --out_base_dir out/$DATA --load_path $PRETR_DIR/save/$PRETR_FROM-$STEP --shared_path $PRETR_DIR/shared.json --load_trained_model --run_id $RUN_ID $ADD
elif [ "$TASK" = "test" ]
then
	python -m basic.cli --mode test --data_dir $DATA_DIR --out_base_dir out/$DATA --shared_path $PRETR_FROM/shared.json --run_id $RUN_ID --load_step $STEP $ADD
else
	echo "WRONG TASK. [finetune | test]"
fi


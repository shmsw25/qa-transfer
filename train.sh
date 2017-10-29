#! /bin/bash

DATA=$1
TASK=$2
RUN_ID=$3
PRETR_FROM=$4
STEP=$5

$PRETR_RUN_ID=00
PRETR_DIR=out/squad/$PRETR_FROM/$PRETR_RUN_ID
	
if [ $DATA = "wikiqa" ]
then
	python -m wikiqa.prepro-class
	DATA_DIR="data/wikiqa-class"
	OUT_DIR="out/wikiqa"
	LOAD_SHARED="False"
	CLASSIFIER="maxpool"
elif [ $DATA = "semeval" ]
then
	python -m semeval.prepro
	DATA_DIR="data/semeval"
	OUT_DIR="out/semeval"
	LOAD_SHARED="True"
	CLASSIFIER="sumpool"
else
	echo "WRONG DATA. [wikiqa | semeval]"
	exit
fi

if [ $TASK = "finetune" ]
then
	CUDA_VISIBLE_DEVICES=2 python -m basic.cli --data_dir $DATA_DIR --out_base_dir $OUT_DIR --num_steps 5000 --classifier $CLASSIFIER --load_path $PRETR_DIR/save/$PRETR_FROM-$STEP --shared_path $PRETR_DIR/shared.json --load_trained_model --run_id $RUN_ID --load_shared $LOAD_SHARED
elif [ "$TASK" = "test" ]
then
	CUDA_VISIBLE_DEVICES=1 python -m basic.cli --mode test --data_dir $DATA_DIR --out_base_dir $OUT_DIR --classifier $CLASSIFIER --shared_path $PRETR_FROM/shared.json --run_id $RUN_ID --load_step $STEP
else
	echo "WRONG TASK. [finetune | test]"
fi


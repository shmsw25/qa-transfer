#! /bin/bash

if [ "$1" = "span" ]
then
	python -m basic.cli --model_name basic --data_dir data/squad --out_base_dir out/squad --noload --num_steps 20000 --dev_name dev --eval_period 500 --save_period 500
	python -m basic.cli --mode test --model_name basic --data_dir data/squad --out_base_dir out/squad --test_name dev
elif [ "$1" = "class" ]
then
	python -m basic.cli --data_dir data/squad-class --out_base_dir out/squad --noload --num_steps 40000 --dev_name dev --eval_period 500 --save_period 500
	python -m basic.cli --mode test --data_dir data/squad-class --out_base_dir out/squad --test_name dev
fi


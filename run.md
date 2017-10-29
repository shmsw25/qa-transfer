## Question Answering through Transfer Learning


- This is the original implementation of "Question Answering through Transfer Learning from Large Fine-grained Supervision Data". [[paper](http://aclweb.org/anthology/P17-2081)] [[poster](https://shmsw25.github.io/assets/acl2017_poster.pdf)]
- Most parts were adapted & modified from "Bi-directional Attention Flow". [[paper](https://arxiv.org/pdf/1611.01603.pdf)] [[code](https://github.com/allenai/bi-att-flow)]
- Evaluation scripts for SemEval were adapted & modified from [SemEval-2016 official scorer](http://alt.qcri.org/semeval2016/task3/index.php?id=data-and-tools).
- Please contact [Sewon Min](https://shmsw25.github.io) ([email](mailto:shmsw25@snu.ac.kr)) for questions and suggestions.

Codes include

- pretraining BiDAF (span-level QA) and BiDAF-T (sentence-level QA) on SQuAD
- training on WikiQA
- training on SemEval-2016 (Task 3A)

### 0. Download & Preprocessing Data

First, download data (SQuAD, WikiQA, SemEval-2016, GLoVe, NLTK). This will download files to $HOME/data
```
chmod +x download.sh; ./download.sh
```

Then, preprocess data.
```
python -m squad.prepro 			# for sqaud in span-level QA setting
python -m squad.convert2class
python -m squad.prepro-class 	# for squad in sentence-level QA setting
python -m wikiqa.prepro-class	# for wikiqa
python -m semeval.prepro		# for semeval
```

### 1. Pretraining

2.1 Pretraining BiDAF for Span-level QA
```
python -m basic.cli --model_name basic --data_dir data/squad --out_base_dir out/squad --noload --num_steps 20000 --dev_name dev
python -m basic.cli --mode test --model_name basic --data_dir data/squad --out_base_dir out/squad --test_name dev --load_step <LOAD_STEP>
```
Each line is for training and testing, respectively.

- use `run_id` flag, if you want to run multiple times (Default = 00)
- `out_base_dir` is for storing saves, logs, evals and shared.json. These are stored in `<out_base_dir>/basic/<run_id>`.
- `num_steps` is number of training steps (global_steps). 20000 is recommended for BiDAF.
- `dev_name` and `test_name` are 'dev' or 'test'. (Default='test') We use 'dev' for SQuAD because test set of SQuAD is not available.

For tensorboard, run `tensorboard --logdir=out/squad/basic/<RUN_ID>/log`

2.2 Pretraining BiDAF-T for Sentence-level QA
```
python -m basic.cli --data_dir data/squad-class --out_base_dir out/squad --noload --num_steps 40000 --dev_name dev
python -m basic.cli --mode test --data_dir data/squad-class --out_base_dir out/squad --test_name dev --load_step <LOAD_STEP>
```

- `num_steps`: 40000 is recommended for BiDAF-T.

Note: Accuracy printed during training and testing is for each sentence, so it does not mean accuracy for each (context, question) pair.


### 2. Training & Testing
2.1. WikiQA

Basically, it use pretrained model from SQuAD.
```
python -m basic.cli --data_dir data/wikiqa-class --out_base_dir out/wikiqa --num_steps 5000 --load_path <LOAD_PATH> --shared_path <SHARED_PATH> --load_trained_model --dev_name dev
python -m basic.cli --mode test -data_dir data/wikiqa-class --out_base_dir out/wikiqa --shared_path <SHARED_PATH> --load_step <LOAD_STEP>
```

- `load_path` is path of saved pretrained model. ex) `out/squad/basic/00/save`
- `shared_path` is path of shared.json of pretrained model. ex) `out/squad/basic/00/shared.json`
- `load_trained_model` should be true in order to use pretrained model. If you do not want to use pretrained model, you can omit this, along with `load_path` and `shared_path`.
- `dev_name` and `test_name` is `dev` or `test`.


Checking Precision, Recall and P@1 in a real time during training is not implemented yet. Here is how to evaluate on these metrics.

First, choose global_step which has the highest metric. (In the paper, we choose one with the highest Precision, following previous works)
```
python -m wikiqa.result --run_ids XX --start_step YY --end_step ZZ --eval_name dev
```
Then, evaluate on test set.
```
python -m wikiqa.result --run_ids XX --global_steps YY
```
You can run evaluation for multiple models with different run_ids. Use `run_ids` and `steps`, separated with commas. For example,
```
python -m wikiqa.result --run_ids 00,01,02 --steps 19000,20000,21000
```
will evaluate 3 models with `run_id 00 step 19000`, `run_id 01 step 20000` and `run_id 02 step 21000`.

For ensemble model, please run as follows.
```
python -m wikiqa.result --run_ids 00,01,02 --steps 19000,20000,21000 --ensemble
```
Note: The ensemble model in the paper use 12 different pretrained models on SQuAD.

2.2 SemEval

Basically, it use pretrained model from SQuAD.
```
python -m basic.cli --data_dir data/semeval --out_base_dir out/semeval --num_steps 5000 --load_path <LOAD_PATH> --shared_path <SHARED_PATH> --load_shared --load_trained_model
python -m basic.cli --mode test --data_dir data/semeval --out_base_dir out/semeval --shared_path <SHARED_PATH> --load_step <LOAD_STEP> --load_shared
```

Then, evaluate the model.
```
python -m semeval.result --run_ids XX --start_step YY --end_step ZZ
```
This will store your answer in `semeval/store` for SemEval-2016 scorer. The scorer was written in Python2.7.
```
python2.7 semeval/evaluation/MAP_scripts/ev.py semeval_store/test-gold semeval/store/test-{run_id}-{global_step(in 6 digits)}
```



### 3. Results

3.1 WikiQA

| Pretrained | MAP | MRR | P@1 | 
| ---------- |:---:|:---:|:---:| 
| SQuAD-T    |76.44|77.85|64.61|
| SQuAD      |79.90|82.01|70.37|
| SQuAD\*    |83.20|84.58|75.31|

3.2 SemEval-2016

| Pretrained | MAP | MRR | P@1 |
| ---------- |:---:|:---:|:---:|
| SQuAD-T    |76.30|82.51|86.64|
| SQuAD      |78.37|85.58|87.68|
| SQuAD\*    |80.20|86.44|89.14|




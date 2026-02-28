export VULNSC_MODEL_ROOT=/disk1/hs/model
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1


### cb

cd CodeBERT

CUDA_VISIBLE_DEVICES=2 python run.py \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--train_data_file=../../data/enhance/devign/gpt4o/0/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/0/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/0/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=2 python run.py \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_eval \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/0/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/0/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/0/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457 2>&1 | tee test.log

python evaluator.py -a ../../data/enhance/devign/gpt4o/0/test.jsonl -p ./saved_models/predictions.txt -m "cb" -e "none-enhance" -t "none" -s "123457"


CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--train_data_file=../../data/enhance/devign/gpt4o/0/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/0/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/0/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_eval \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/0/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/0/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/0/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457 2>&1 | tee test.log

python evaluator.py -a ../../data/enhance/devign/gpt4o/0/test.jsonl -p ./saved_models/predictions.txt -m "cb" -e "enhance" -t "basic" -s "123457"




CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--train_data_file=../../data/enhance/devign/gpt4o/1/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/1/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/1/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_eval \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/1/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/1/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/1/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457 2>&1 | tee test.log

python evaluator.py -a ../../data/enhance/devign/gpt4o/1/test.jsonl -p ./saved_models/predictions.txt -m "cb" -e "enhance" -t "behavior" -s "123457"



CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--train_data_file=../../data/enhance/devign/gpt4o/2/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/2/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/2/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_eval \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/2/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/2/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/2/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457 2>&1 | tee test.log

python evaluator.py -a ../../data/enhance/devign/gpt4o/2/test.jsonl -p ./saved_models/predictions.txt -m "cb" -e "enhance" -t "oneshot" -s "123457"


CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--train_data_file=../../data/enhance/devign/gpt4o/3/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/3/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/3/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_eval \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/3/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/3/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/3/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457 2>&1 | tee test.log

python evaluator.py -a ../../data/enhance/devign/gpt4o/3/test.jsonl -p ./saved_models/predictions.txt -m "cb" -e "enhance" -t "cot" -s "123457"


### gcb

cd ../GraphCodeBERT

CUDA_VISIBLE_DEVICES=2 python run.py \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/graphcodebert-base  \
--model_name_or_path=microsoft/graphcodebert-base  \
--do_train \
--train_data_file=../../data/enhance/devign/gpt4o/0/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/0/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/0/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=2 python run.py \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/graphcodebert-base  \
--model_name_or_path=microsoft/graphcodebert-base  \
--do_eval \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/0/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/0/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/0/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457 2>&1 | tee test.log

python evaluator.py -a ../../data/enhance/devign/gpt4o/0/test.jsonl -p ./saved_models/predictions.txt -m "gcb" -e "none-enhance" -t "none" -s "123457"


CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/graphcodebert-base  \
--model_name_or_path=microsoft/graphcodebert-base  \
--do_train \
--train_data_file=../../data/enhance/devign/gpt4o/0/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/0/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/0/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/graphcodebert-base  \
--model_name_or_path=microsoft/graphcodebert-base  \
--do_eval \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/0/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/0/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/0/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457 2>&1 | tee test.log

python evaluator.py -a ../../data/enhance/devign/gpt4o/0/test.jsonl -p ./saved_models/predictions.txt -m "gcb" -e "enhance" -t "basic" -s "123457"




CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/graphcodebert-base  \
--model_name_or_path=microsoft/graphcodebert-base  \
--do_train \
--train_data_file=../../data/enhance/devign/gpt4o/1/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/1/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/1/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/graphcodebert-base  \
--model_name_or_path=microsoft/graphcodebert-base  \
--do_eval \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/1/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/1/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/1/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457 2>&1 | tee test.log

python evaluator.py -a ../../data/enhance/devign/gpt4o/1/test.jsonl -p ./saved_models/predictions.txt -m "gcb" -e "enhance" -t "behavior" -s "123457"



CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/graphcodebert-base  \
--model_name_or_path=microsoft/graphcodebert-base  \
--do_train \
--train_data_file=../../data/enhance/devign/gpt4o/2/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/2/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/2/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/graphcodebert-base  \
--model_name_or_path=microsoft/graphcodebert-base  \
--do_eval \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/2/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/2/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/2/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457 2>&1 | tee test.log

python evaluator.py -a ../../data/enhance/devign/gpt4o/2/test.jsonl -p ./saved_models/predictions.txt -m "gcb" -e "enhance" -t "oneshot" -s "123457"


CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/graphcodebert-base  \
--model_name_or_path=microsoft/graphcodebert-base  \
--do_train \
--train_data_file=../../data/enhance/devign/gpt4o/3/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/3/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/3/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/graphcodebert-base  \
--model_name_or_path=microsoft/graphcodebert-base  \
--do_eval \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/3/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/3/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/3/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457 2>&1 | tee test.log

python evaluator.py -a ../../data/enhance/devign/gpt4o/3/test.jsonl -p ./saved_models/predictions.txt -m "gcb" -e "enhance" -t "cot" -s "123457"



### ux

cd ../UniXcoder

CUDA_VISIBLE_DEVICES=2 python run.py \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/unixcoder-base \
--model_name_or_path=microsoft/unixcoder-base \
--do_train \
--train_data_file=../../data/enhance/devign/gpt4o/0/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/0/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/0/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=2 python run.py \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/unixcoder-base \
--model_name_or_path=microsoft/unixcoder-base \
--do_eval \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/0/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/0/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/0/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457 2>&1 | tee test.log

python evaluator.py -a ../../data/enhance/devign/gpt4o/0/test.jsonl -p ./saved_models/predictions.txt -m "ux" -e "none-enhance" -t "none" -s "123457"


CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/unixcoder-base \
--model_name_or_path=microsoft/unixcoder-base \
--do_train \
--train_data_file=../../data/enhance/devign/gpt4o/0/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/0/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/0/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/unixcoder-base \
--model_name_or_path=microsoft/unixcoder-base \
--do_eval \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/0/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/0/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/0/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457 2>&1 | tee test.log

python evaluator.py -a ../../data/enhance/devign/gpt4o/0/test.jsonl -p ./saved_models/predictions.txt -m "ux" -e "enhance" -t "basic" -s "123457"




CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/unixcoder-base \
--model_name_or_path=microsoft/unixcoder-base \
--do_train \
--train_data_file=../../data/enhance/devign/gpt4o/1/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/1/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/1/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/unixcoder-base \
--model_name_or_path=microsoft/unixcoder-base \
--do_eval \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/1/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/1/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/1/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457 2>&1 | tee test.log

python evaluator.py -a ../../data/enhance/devign/gpt4o/1/test.jsonl -p ./saved_models/predictions.txt -m "ux" -e "enhance" -t "behavior" -s "123457"



CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/unixcoder-base \
--model_name_or_path=microsoft/unixcoder-base \
--do_train \
--train_data_file=../../data/enhance/devign/gpt4o/2/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/2/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/2/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/unixcoder-base \
--model_name_or_path=microsoft/unixcoder-base \
--do_eval \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/2/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/2/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/2/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457 2>&1 | tee test.log

python evaluator.py -a ../../data/enhance/devign/gpt4o/2/test.jsonl -p ./saved_models/predictions.txt -m "ux" -e "enhance" -t "oneshot" -s "123457"


CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/unixcoder-base \
--model_name_or_path=microsoft/unixcoder-base \
--do_train \
--train_data_file=../../data/enhance/devign/gpt4o/3/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/3/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/3/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=2 python run.py \
--enhance \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/unixcoder-base \
--model_name_or_path=microsoft/unixcoder-base \
--do_eval \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/3/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/3/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/3/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 123457 2>&1 | tee test.log

python evaluator.py -a ../../data/enhance/devign/gpt4o/3/test.jsonl -p ./saved_models/predictions.txt -m "ux" -e "enhance" -t "cot" -s "123457"



# lv

cd ../LineVul

CUDA_VISIBLE_DEVICES=2 python linevul_main.py \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/0/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/0/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/0/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--method=linevul \
--enhance=none-enhance \
--prompt=none \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log \


CUDA_VISIBLE_DEVICES=2 python linevul_main.py \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/0/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/0/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/0/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--method=linevul \
--enhance=enhance \
--prompt=basic \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log \


CUDA_VISIBLE_DEVICES=2 python linevul_main.py \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/1/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/1/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/1/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--method=linevul \
--enhance=enhance \
--prompt=behavior \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log \

CUDA_VISIBLE_DEVICES=2 python linevul_main.py \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/2/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/2/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/2/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--method=linevul \
--enhance=enhance \
--prompt=oneshot \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log \

CUDA_VISIBLE_DEVICES=2 python linevul_main.py \
--output_dir=./saved_models \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--do_test \
--train_data_file=../../data/enhance/devign/gpt4o/3/train.jsonl \
--eval_data_file=../../data/enhance/devign/gpt4o/3/valid.jsonl \
--test_data_file=../../data/enhance/devign/gpt4o/3/test.jsonl \
--metric=eval_f1 \
--epoch 3 \
--block_size 500 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--method=linevul \
--enhance=enhance \
--prompt=cot \
--evaluate_during_training \
--seed 123457  2>&1 | tee train.log \

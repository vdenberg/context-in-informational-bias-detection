# run lm on train.txt from dont-stop-pretraining/


LM_DIR = '../basil_and_source_tapt/data/lm/'
# basil tapt
rm pretrained_models/roberta_basil_tapt && mkdir pretrained_models/roberta_basil_tapt
/opt/slurm/bin/srun --partition kama --gres=gpu:1 --mem 20GB python -m scripts.run_language_modeling \
                                        --line_by_line \
                                        --model_type roberta-base \
                                        --tokenizer_name roberta-base \
                                        --mlm \
                                        --per_gpu_train_batch_size 12 \
                                        --gradient_accumulation_steps 256  \
                                        --model_name_or_path roberta-base \
                                        --do_eval \
                                        --evaluate_during_training  \
                                        --do_train \
                                        --num_train_epochs 100  \
                                        --learning_rate 0.0001 \
                                        --logging_steps 50 \
                                        --seed 11 \
                                        --train_data_file $LM_DIR'basil_train.txt' \
                                        --eval_data_file $LM_DIR'basil_eval.txt' \
                                        --output_dir 'pretrained_models/roberta_basil_tapt' \

# BASELINES
# BERT on BASIL
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters BERT_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir model_logs/basil/bert_sentence_split \
                --dataset basil_sentence_split \
                --model bert-base-cased \
                -x 11 22 33 44 55

# RoBERTa-base on BASIL
ITER=1
MAXITER=11
while [ "$ITER" -lt "$MAXITER" ];
do
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir 'model_logs/basil/roberta_'$ITER \
                --dataset basil_$ITER \
                --model roberta-base \
                -x 11 22 33 44 55
ITER=$((ITER+1));
done

# roberta-TAPT on BASIL
ITER=1
MAXITER=11
while [ "$ITER" -lt "$MAXITER" ];
do
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir model_logs/basil/roberta_$ITER \
                --dataset basil_$ITER \
                --model 'pretrained_models/roberta_basil_tapt' \
                -x 11 22 33 44 55
ITER=$((ITER+1));
done

# basil cur tapt
#todo fix fox_nyt_hpo_cur_train.txt
rm pretrained_models/roberta_basil_tapt && mkdir pretrained_models/roberta_basil_tapt
/opt/slurm/bin/srun --partition kama --gres=gpu:1 --mem 20GB python -m scripts.run_language_modeling \
                                        --line_by_line \
                                        --model_type roberta-base \
                                        --tokenizer_name roberta-base \
                                        --mlm \
                                        --per_gpu_train_batch_size 12 \
                                        --gradient_accumulation_steps 256  \
                                        --model_name_or_path roberta-base \
                                        --do_eval \
                                        --evaluate_during_training  \
                                        --do_train \
                                        --num_train_epochs 100  \
                                        --learning_rate 0.0001 \
                                        --logging_steps 50 \
                                        --seed 11 \
                                        --train_data_file $LM_DIR'fox_nyt_hpo_cur_train.txt' \
                                        --eval_data_file $LM_DIR'basil_eval.txt' \
                                        --output_dir 'pretrained_models/roberta_basil_tapt' \

# CurTAPT on BASIL
ITER=1
MAXITER=11
while [ "$ITER" -lt "$MAXITER" ];
do
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir model_logs/basil/roberta_$ITER \
                --dataset basil_$ITER \
                --model roberta-base \
                -x 11 22 33 44 55
ITER=$((ITER+1));
done
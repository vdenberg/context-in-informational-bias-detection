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

# basil fox tapt
rm pretrained_models/roberta_fox_basil_cur_tapt && mkdir pretrained_models/roberta_fox_basil_cur_tapt
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
                                        --train_data_file $LM_DIR'fox_cur_train.txt' \
                                        --eval_data_file $LM_DIR'fox_basil_eval.txt' \
                                        --output_dir 'pretrained_models/fox_roberta_basil_tapt' \


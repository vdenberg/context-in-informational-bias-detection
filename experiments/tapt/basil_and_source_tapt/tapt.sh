cd ../dont-stop-pretraining
mkdir pretrained_models/roberta_basil_tapt
mkdir pretrained_models/roberta_basil_cur_tapt
mkdir pretrained_models/roberta_basil_plus_cur_tapt

# TAPT pretraining
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
                                        --eval_data_file "../data/lm/basil_eval.txt"  \
                                        --train_data_file "../data/lm/basil_train.txt" \
                                        --output_dir "pretrained_models/roberta_basil_tapt"

# CurTAPT pretraining
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
                                        --eval_data_file ../data/lm/basil_eval.txt \
                                        --train_data_file "../data/lm/fox_nyt_hpo_cur_train.txt" \
                                        --output_dir "pretrained_models/roberta_basil_cur_tapt"

# CurTAPT pretraining
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
                                        --eval_data_file ../data/lm/basil_eval.txt \
                                        --train_data_file "../data/lm/basil_fox_nyt_hpo_cur_train.txt" \
                                        --output_dir "pretrained_models/roberta_basil_plus_cur_tapt" \

# run lm on train.txt
cd ../dont-stop-pretraining
rm pretrained_models/re_roberta_hp_515 && mkdir pretrained_models/re_roberta_hp_515
/opt/slurm/bin/srun --partition kama --gres=gpu:1 --mem 20GB python -m scripts.run_language_modeling \
                                        --train_data_file ../data/hyperpartisan/train.txt \
                                        --line_by_line \
                                        --output_dir pretrained_models/re_roberta_hp_515 \
                                        --model_type roberta-base \
                                        --tokenizer_name roberta-base \
                                        --mlm \
                                        --per_gpu_train_batch_size 12 \
                                        --gradient_accumulation_steps 256  \
                                        --model_name_or_path roberta-base \
                                        --eval_data_file ../data/hyperpartisan/eval.txt \
                                        --do_eval \
                                        --evaluate_during_training  \
                                        --do_train \
                                        --num_train_epochs 100  \
                                        --learning_rate 0.0001 \
                                        --logging_steps 50 \
                                        --seed 11

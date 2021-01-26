cd ../dont-stop-pretraining
/opt/slurm/bin/srun --partition kama --gres=gpu:1 --mem 2GB python -m scripts.run_language_modeling \
                                        --train_data_file ../data/hyperpartisan/docs.txt \
                                        --line_by_line \
                                        --output_dir roberta-tapt \
                                        --model_type roberta-base \
                                        --tokenizer_name roberta-base \
                                        --mlm \
                                        --per_gpu_train_batch_size 16 \
                                        --gradient_accumulation_steps 16  \
                                        --model_name_or_path roberta-base \
                                        --eval_data_file ../data/hyperpartisan/unlabeled/eval.txt \
                                        --do_eval \
                                        --evaluate_during_training  \
                                        --do_train \
                                        --num_train_epochs 100  \
                                        --learning_rate 0.0001 \
                                        --logging_steps 50
mkdir basil_cur_tapt_story_split_ft_results
cd ../dont-stop-pretraining
mkdir pretrained_models/roberta_basil_cur_tapt

# CurTAPT pretraining
#todo fix fox_nyt_hpo_cur_train.txt
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
                                        --train_data_file '../data/lm/fox_nyt_hpo_cur_train.txt' \
                                        --output_dir 'pretrained_models/roberta_basil_cur_tapt' \

# CurTAPT on Story Split
ITER=1
MAXITER=11
while [ "$ITER" -lt "$MAXITER" ];
do
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir '../basil_and_source_tapt/basil_cur_tapt_story_split_ft_results/basil_cur_tapt_'$ITER \
                --dataset basil_$ITER \
                --model 'pretrained_models/roberta_basil_cur_tapt' \
                -x 11 22 33 44 55
ITER=$((ITER+1));
done

python ../eval/agg_eval.py -dir ../basil_and_source_tapt/basil_cur_tapt_story_split_ft_results

mkdir roberta_sentence_split_ft_results
mkdir roberta_story_split_ft_results
cd ../dont-stop-pretraining
# RoBERTa on Sentence Split
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir ../baseline_reproduction/roberta_sentence_split_ft_results/roberta_sentence_split \
                --dataset basil_sentence_split \
                --model roberta-base \
                -x 11 22 33 44 55

# RoBERTa on Story Split
ITER=1
MAXITER=11
while [ "$ITER" -lt "$MAXITER" ];
do
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir '../baseline_reproduction/roberta_story_split_ft_results/roberta_'$ITER \
                --dataset basil_$ITER \
                --model roberta-base \
                -x 11 22 33 44 55
ITER=$((ITER+1));
done

python ../eval/agg_eval.py -dir roberta_sentence_split_ft_results roberta_story_split_ft_results
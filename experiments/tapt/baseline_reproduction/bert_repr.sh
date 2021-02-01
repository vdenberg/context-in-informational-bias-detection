mkdir bert_sentence_split_ft_results
mkdir bert_story_split_ft_results
cd ../dont-stop-pretraining
# BERT on Sentence Split
#/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
#                --hyperparameters BERT_CLASSIFIER_MINI \
#                --config training_config/classifier.jsonnet \
#                --serialization_dir ../baseline_reproduction/bert_sentence_split_ft_results/bert_sentence_split \
#                --dataset basil_sentence_split \
#                --model bert-base-cased \
#                -x 11 22 33 44 55

# BERT on Story Split
ITER=1
MAXITER=11
while [ "$ITER" -lt "$MAXITER" ];
do
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters BERT_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir '../baseline_reproduction/bert_story_split_ft_results/bert_'$ITER \
                --dataset basil_$ITER \
                --model bert-base-cased \
                -x 11 22 33 44 55
ITER=$((ITER+1));
done


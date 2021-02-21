mkdir basil_plus_cur_tapt_story_split_ft_results
cd ../dont-stop-pretraining

# CurTAPT on Story Split
ITER=1
MAXITER=11
while [ "$ITER" -lt "$MAXITER" ];
do
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir '../basil_and_source_tapt/basil_plus_cur_tapt_story_split_ft_results/basil_cur_tapt_'$ITER \
                --dataset basil_$ITER \
                --model 'pretrained_models/roberta_basil_plus_cur_tapt' \
                -x 11 22 33 44 55 66 77 88 99 1010
ITER=$((ITER+1));
done

python ../eval/agg_eval.py -dir ../basil_and_source_tapt/basil_cur_tapt_story_split_ft_results

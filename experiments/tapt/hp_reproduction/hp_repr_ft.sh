ITER=1
MAXITER=6

while [ "$ITER" -lt "$MAXITER" ];
do

echo "===starting "$ITER

SEED=$((ITER*11));
$LOGDIR='model_logs/hyperpartisan_tapt/'$SEED;
mkdir $LOGDIR
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 2GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir $LOGDIR \
                --dataset hyperpartisan_news \
                --model roberta-base \
                -x $SEED

ITER=$((ITER+1));

done
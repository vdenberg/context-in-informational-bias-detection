mkdir model_logs/hyperpartisan_tapt/
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 2GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir model_logs/hyperpartisan_tapt/ \
                --dataset hyperpartisan_news \
                --model roberta-base \
                --jackknife \
                -x 11 22 33 44 55
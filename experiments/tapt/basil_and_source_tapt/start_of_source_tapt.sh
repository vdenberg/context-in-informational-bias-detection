# BERT on BASIL
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters BERT_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir model_logs/basil/bert_sentence_split \
                --dataset basil_sentence_split \
                --model bert-base-cased \
                -x 11 22 33 44 55

# RoBERTa-base on BASIL
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir model_logs/basil/roberta_sentence_split \
                --dataset basil_sentence_split \
                --model roberta-base \
                -x 11

# FOX-TAPT on BASIL
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir model_logs/basil/fox_tapt_1 \
                --dataset basil_1 \
                --model roberta_fox_tapt \
                -x 11


# RoBERTa-base on FOX-BASIL
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir model_logs/fox_basil/roberta_1 \
                --dataset fox_basil_1 \
                --model bert-base-uncased \
                -x 11

# FOX-TAPT on FOX-BASIL
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir model_logs/fox_basil/fox_tapt_1 \
                --dataset fox_basil_1 \
                --model roberta_fox_tapt  \
                -x 11
wget https://zenodo.org/record/1489920/files/articles-training-bypublisher-20181122.zip?download=1
mv articles-training-bypublisher-20181122.zip?download=1 articles-training-bypublisher-20181122.zip
unzip articles-training-bypublisher-20181122.zip && rm articles-training-bypublisher-20181122.zip
/opt/slurm/bin/srun --partition compute --mem 20GB python hp_preprocess.py

cd ../dont-stop-pretraining
rm pretrained_models/re_hp_515 && mkdir pretrained_models/re_hp_515
/opt/slurm/bin/srun --partition kama --gres=gpu:1 --mem 20GB python -m scripts.run_language_modeling \
                                        --train_data_file ../data/hyperpartisan/train.txt \
                                        --line_by_line \
                                        --output_dir pretrained_models/re_hp_515 \
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

rm pretrained_models/re_hp_5000 && mkdir pretrained_models/re_hp_5000
/opt/slurm/bin/srun --partition kama --gres=gpu:1 --mem 20GB python -m scripts.run_language_modeling \
                                        --train_data_file ../data/hyperpartisan/curated.txt \
                                        --line_by_line \
                                        --output_dir pretrained_models/re_hp_5000\
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

mkdir model_logs/re_hp_515_ft/
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 2GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir model_logs/hyperpartisan_tapt/ \
                --dataset hyperpartisan_news \
                --model roberta-base \
                --jackknife \
                -x 11 22 33 44 55

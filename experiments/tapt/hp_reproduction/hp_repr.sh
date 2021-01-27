# start in experiments/tapt/hp_reproduction

# TAPT

# preprocess
/opt/slurm/bin/srun --partition compute --mem 20GB python hp_preprocess.py -labeled

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

# ft reproduced lm on jsonls
mkdir ../hp_reproduction/re_roberta_hp_515_ft_results/results
rm -r ../hp_reproduction/re_roberta_hp_515_ft_results/results*
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI --config training_config/classifier.jsonnet \
                --serialization_dir ../hp_reproduction/re_roberta_hp_515_ft_results/results \
                --dataset hyperpartisan_news \
                --model $(pwd)/pretrained_models/dsp_roberta_base_dapt_news_tapt_hyperpartisan_news_515 \
                --seed 11 22 33 44 55 --jackknife

# CURATED TAPT

# collect curated data
wget https://zenodo.org/record/1489920/files/articles-training-bypublisher-20181122.zip?download=1
mv articles-training-bypublisher-20181122.zip?download=1 articles-training-bypublisher-20181122.zip
unzip articles-training-bypublisher-20181122.zip && rm articles-training-bypublisher-20181122.zip

# preprocess
/opt/slurm/bin/srun --partition compute --mem 20GB python hp_preprocess.py -curated

rm pretrained_models/re_roberta_hp_5000 && mkdir pretrained_models/re_roberta_hp_5000
/opt/slurm/bin/srun --partition kama --gres=gpu:1 --mem 20GB python -m scripts.run_language_modeling \
                                        --train_data_file ../data/hyperpartisan/curated.txt \
                                        --line_by_line \
                                        --output_dir pretrained_models/re_roberta_hp_5000\
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

mkdir ../hp_reproduction/re_roberta_hp_5000_ft_results/
/opt/slurm/bin/srun --partition kama --gres=gpu:1  --mem 20GB python -m scripts.train --device 0 --perf +f1 --evaluate_on_test \
                --hyperparameters ROBERTA_CLASSIFIER_MINI \
                --config training_config/classifier.jsonnet \
                --serialization_dir ../hp_reproduction/re_roberta_hp_5000_ft_results/ \
                --dataset hyperpartisan_news \
                --model $(pwd)/pretrained_models/re_roberta_hp_5000 \
                --jackknife \
                --seed 11 22 33 44 55

python -m scripts.download_model --model allenai/dsp_roberta_base_dapt_news_tapt_hyperpartisan_news_515 \
        --serialization_dir $(pwd)/pretrained_models/dsp_roberta_base_dapt_news_tapt_hyperpartisan_news_515

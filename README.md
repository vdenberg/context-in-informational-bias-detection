# Context In Informational Bias Detection

This repository contains code and instructions for
- finetuning the PLM's BERT and RoBERTa
- performing Sequential Sentence Classification
- obtaining embeddings from a PLM for further experimentation
- training a Context-Inclusive Model
- further pre-training of RoBERTa on the BASIL corpus of lexical and informational bias towards entities.

For any questions, please contact esthervdenberg [at] gmail.com.

### Citing

This repository documents the experiments in a paper on the automatic detection of informational bias towards entities 
with neural approaches that take into account context beyond the sentence. 

```bibtex
@inproceedings{berg2020context,
 author = {Esther van den Berg and Katja Markert},
 title = {Context in Informational Bias Detection},
 year = {2020},
 booktitle = {Proceedings of COLING},
}
```

### Installing

1. Clone this repository, ensure you are in an environment with Python 3.7, and install dependencies, including the appropriate cudaversion for PyTorch (original experiments used 10.1):

   ```shell script
   git clone https://github.com/vdenberg/context-in-informational-bias-detection.git
   cd context-in-informational-bias-detection
   conda create -n ciib python=3.7
   conda activate ciib
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm   
   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
   ```
   
    Also make sure your the project directory is included in the pythonpath:
   
    ```shell script
    export PYTHONPATH="/path/to/context-in-informational-bias-detection"
    ```
  

2. Download and unzip BASIL Corpus from: https://github.com/marshallwhiteorg/emnlp19-media-bias/blob/master/emnlp19-BASIL.zip
   to **context-in-informational-bias-detection/data/emnlp19-BASIL**

3. Preprocess data.

```shell script
    sh preprocess.sh
   ```

4. Choose the experiment you're interested in, and continue instructions below, or, to reproduce the analyses from the paper,
run the following:

 ```shell script
    python analyses/significance_tests.py
    python analyses/performance_analysis.py
   ```

### BERT and RoBERTa baselines

 ```shell script
    # sentence classification with bert
    python experiments/finetune_plm.py -clf_task sent_clf -model bert

    # sentence classification with roberta
    python experiments/finetune_plm.py -clf_task sent_clf -model rob_base

    # token classification with bert
    python experiments/finetune_plm.py -clf_task tok_clf -model bert

    # token classification with roberta
    python experiments/finetune_plm.py -clf_task tok_clf -model rob_base
   ```

### Sequential Sentence Classification

```shell script
# SSC without window with sequence length of 5 
python experiments/finetune_plm.py -clf_task seq_sent_clf -seq_len 5 

# Window SSC with sequence length of 10
python experiments/finetune_plm.py -clf_task seq_sent_clf -seq_len 10 -win
```

### Article and Event Context

```shell script
# prepare embeddings
python experiments/finetune_plm.py -clf_task sent_clf -model rob_base -sv 49 -embeds

# ArtCIM
python experiments/context_inclusive.py -context art -cim_type cim

# ArtCIM*
python experiments/context_inclusive.py -context art -cim_type cim*

# EvCIM
python experiments/context_inclusive.py -context ev -cim_type cim

# EvCIM*
python experiments/context_inclusive.py -context ev -cim_type cim*
```

### Domain Context

1. Clone https://github.com/allenai/dont-stop-pretraining into experiments directory

2. Follow install instructions of https://github.com/allenai/dont-stop-pretraining.

3. Run following command to get basil-adapted models:

    ##### TAPT
    ```shell script
    python -m scripts.run_language_modeling --train_data_file ../../data/inputs/tapt/basil_train.txt \
                                            --line_by_line \
                                            --output_dir roberta-basil-tapt \
                                            --model_type roberta-base \
                                            --tokenizer_name roberta-base \
                                            --mlm \
                                            --per_gpu_train_batch_size 6 \
                                            --gradient_accumulation_steps 6  \
                                            --model_name_or_path roberta-base \
                                            --do_eval \
                                            --eval_data_file ../../data/inputs/tapt/basil_test.txt \
                                            --evaluate_during_training  \
                                            --do_train \
                                            --num_train_epochs 100  \
                                            --learning_rate 0.0001 \
                                            --logging_steps 50
    ```
    ##### DAPTTAPT
    ```shell script
    python -m scripts.run_language_modeling --train_data_file ../../data/inputs/tapt/basil_train.txt \
                                            --line_by_line \
                                            --output_dir roberta-basil-dapttapt \
                                            --model_type roberta-base \
                                            --tokenizer_name roberta-base \
                                            --mlm \
                                            --per_gpu_train_batch_size 6 \
                                            --gradient_accumulation_steps 6  \
                                            --model_name_or_path ../pretrained_models/news_roberta_base \
                                            --do_eval \
                                            --eval_data_file ../../data/inputs/tapt/basil_test.txt \
                                            --evaluate_during_training  \
                                            --do_train \
                                            --num_train_epochs 100  \
                                            --learning_rate 0.0001 \
                                            --logging_steps 50
    ```
            
3. Run following commands to get source-adapted models: 
(For DAPTTAPT specify ```--output_dir  roberta-fox-daptapt``` and ```--model_name_or_path ../pretrained_models/news_roberta_base```)

    ```shell script
    python -m scripts.run_language_modeling --train_data_file ../../data/inputs/tapt/fox_train.txt \
                                            --line_by_line \
                                            --output_dir roberta-fox-tapt \
                                            --model_type roberta-base \
                                            --tokenizer_name roberta-base \
                                            --mlm \
                                            --per_gpu_train_batch_size 6 \
                                            --gradient_accumulation_steps 6  \
                                            --model_name_or_path roberta-base \
                                            --do_eval \
                                            --eval_data_file ../../data/inputs/tapt/basil_fox_test.txt \
                                            --evaluate_during_training  \
                                            --do_train \
                                            --num_train_epochs 150  \
                                            --learning_rate 0.0001 \
                                            --logging_steps 50 \ 
                                            --save_total_limit 2 \
                                            --overwrite_output_dir
    ```
   --overwrite_output_dir
   --should_continue
   
   python -m scripts.train \
        --config training_config/classifier.jsonnet \
        --serialization_dir model_logs/hyperpartisan_base \
        --hyperparameters ROBERTA_CLASSIFIER_SMALL \
        --dataset hyperpartisan_news \
        --model roberta-base \
        --device 0 \
        --perf +f1 \
        --evaluate_on_test
# Context In Informational Bias Detection

### Citing


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
# SSC without window (sequence length of 5 or 10)
python experiments/finetune_plm.py -clf_task seq_sent_clf -seq_len [5|10] 
# Window SSC (sequence length of 5 or 10)
python experiments/finetune_plm.py -clf_task seq_sent_clf -seq_len [5|10] -win
```

### Article and Event Context

```shell script
python experiments/context_inclusive.py -context [art|ev] -cim_type [cim|cim*]
```

### Domain Context

1. Clone https://github.com/allenai/dont-stop-pretraining into experiments directory

2. Follow install instructions of https://github.com/allenai/dont-stop-pretraining.

3. Run following commands to get basil-adapted models:

    ##### TAPT
    ```shell script
    python -m scripts.run_language_modeling --train_data_file data/inputs/tapt/basil_train.txt \
                                            --line_by_line \
                                            --output_dir roberta-basil-tapt \
                                            --model_type roberta-base \
                                            --tokenizer_name roberta-base \
                                            --mlm \
                                            --per_gpu_train_batch_size 6 \
                                            --gradient_accumulation_steps 6  \
                                            --model_name_or_path roberta-base \
                                            --do_eval \
                                            --eval_data_file data/inputs/tapt/basil_test.txt \
                                            --evaluate_during_training  \
                                            --do_train \
                                            --num_train_epochs 100  \
                                            --learning_rate 0.0001 \
                                            --logging_steps 50
    ```
    ##### DAPTTAPT
    ```shell script
    python -m scripts.run_language_modeling --train_data_file data/inputs/tapt/basil_train.txt \
                                            --line_by_line \
                                            --output_dir roberta-basil-dapttapt \
                                            --model_type roberta-base \
                                            --tokenizer_name roberta-base \
                                            --mlm \
                                            --per_gpu_train_batch_size 6 \
                                            --gradient_accumulation_steps 6  \
                                            --model_name_or_path ../pretrained_models/news_roberta_base \
                                            --do_eval \
                                            --eval_data_file data/inputs/tapt/basil_test.txt \
                                            --evaluate_during_training  \
                                            --do_train \
                                            --num_train_epochs 100  \
                                            --learning_rate 0.0001 \
                                            --logging_steps 50
    ```
            

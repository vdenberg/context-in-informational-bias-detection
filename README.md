# Context In Informational Bias Detection

### Citing


```bibtex
@inproceedings{dontstoppretraining2020,
 author = {Suchin Gururangan and Ana Marasović and Swabha Swayamdipta and Kyle Lo and Iz Beltagy and Doug Downey and Noah A. Smith},
 title = {Don't Stop Pretraining: Adapt Language Models to Domains and Tasks},
 year = {2020},
 booktitle = {Proceedings of ACL},
}
```

### Installing

1. Clone this repository and install dependencies:

   ```shell script
   git clone https://github.com/vdenberg/context-in-informational-bias-detection.git
   pip install -r requirements.txt
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
    python experiments/finetune_plm.py 
    python experiments/context_inclusive.py -context ev -cim_type cim
    python analyses/sign_test.py
    python analyses/in_depth_analysis.py
   ```

### BERT and RoBERTa baselines: 

 ```shell script
    python experiments/finetune_plm.py -clf_task [sent_clf|tok_clf] -model [bert|rob_base]
   ```

### Sequential Sentence Classification: 

```shell script
python experiments/finetune_plm.py -clf_task seq_sent_clf -exlen [5|10]] -win
```

### Article & Context: 

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
            

# Context In Informational Bias Detection

### Installing

1. Download & Unzip BASIL Corpus into ```data``` from:
github.com/marshallwhiteorg/emnlp19-media-bias/blob/master/emnlp19-BASIL.zip

2. Ensure installation of Pytorch.

3. Run preprocessing script.

    ```shell script
    sh preprocess.sh
   ```

4. Choose the experiment you're interested in, and continue instructions.

- Sentence Classification & Token Classification 
- Sentence Classification - BERT: sent_clf/finetune_bert.py
- Sentence Classification - RoBERTa: sent_clf/finetune_roberta.py
- Token Classification - BERT: tok_clf/finetune_bert_for_tokclf.py
- Token Classification - RoBERTa: tok_clf/finetune_roberta_for_tokclf.py

### Sequential Sentence Classification: 

seq_sent_clf/seq_sent_clf.py

### Article & Context: 

sent_clf/context_inclusive_model.py

### Domain Context

1. Follow instructions of https://github.com/allenai/dont-stop-pretraining.

2. Run following commands.

    ##### TAPT
    ```
    /opt/slurm/bin/srun --partition gpushort --gres=gpu:1 --mem 20GB python -m scripts.run_language_modeling --train_data_file data/inputs/tapt/basil_train.txt \
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
    ```
    /opt/slurm/bin/srun --partition gpushort --gres=gpu:1 --mem 20GB python -m scripts.run_language_modeling --train_data_file data/inputs/tapt/basil_train.txt \
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
            

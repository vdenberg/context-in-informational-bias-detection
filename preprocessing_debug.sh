touch data/inputs/tapt/basil_train.tsv
touch data/inputs/tapt/basil_test.tsv
python3 preprocessing/preprocessing_for_cim_and_tapt.py # --add_use --add_sbert
python3 preprocessing/preprocessing_for_plm.py --clf_task seq_sent_clf -plm roberta -seqlen 10 -win

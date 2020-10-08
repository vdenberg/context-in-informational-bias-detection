# GENERAL
# python -m spacy download en_core_web_sm

# BASELINES

# sentence classification, BERT
python3 preprocessing/preprocessing_for_plm.py --clf_task sent_clf -plm bert

# sentence classification, RoBERTa
python3 preprocessing/preprocessing_for_plm.py --clf_task sent_clf -plm roberta

# baseline, token classification, BERT
python3 preprocessing/preprocessing_for_plm.py --clf_task tok_clf -plm bert
# baseline, token classification, RoBERTa
python3 preprocessing/preprocessing_for_plm.py --clf_task tok_clf -plm roberta

# SEQUENTIAL SENTENCE CLASSIFICATION

# seq_sent_clf, len 5
python3 preprocessing/preprocessing_for_plm.py --clf_task seq_sent_clf --sequence_length 5 --windowed
python3 preprocessing/preprocessing_for_plm.py --clf_task seq_sent_clf --sequence_length 5
# seq_sent_clf, len 10
python3 preprocessing/preprocessing_for_plm.py --clf_task seq_sent_clf --sequence_length 10 --windowed
python3 preprocessing/preprocessing_for_plm.py --clf_task seq_sent_clf --sequence_length 10

# CIM & DOMAIN
touch data/inputs/tapt/basil_train.tsv
touch data/inputs/tapt/basil_test.tsv
python3 preprocessing/preprocessing_for_cim_and_tapt.py # --add_use --add_sbert




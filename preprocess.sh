# GENERAL
# python -m spacy download en_core_web_sm

# CIM
python3 preprocessing/preprocess_for_cim.py # --add_use --add_sbert

# DOMAIN
python3 preprocessing/preprocess_for_tapt.py

# BASELINES
# sentence classification, BERT
#todo check out why length of preprocessed features is not the same as length of basil
python3 preprocessing/preprocess_for_plm.py --clf_task sent_clf -plm bert

# sentence classification, RoBERTa, three sources
python3 preprocessing/preprocess_for_plm.py --clf_task sent_clf -plm roberta
python3 preprocessing/preprocess_for_plm.py --clf_task sent_clf -plm roberta -source fox
python3 preprocessing/preprocess_for_plm.py --clf_task sent_clf -plm roberta -source nyt
python3 preprocessing/preprocess_for_plm.py --clf_task sent_clf -plm roberta -source hpo

# baseline, token classification, BERT
python3 preprocessing/preprocess_for_plm.py --clf_task tok_clf -plm bert
# baseline, token classification, RoBERTa
python3 preprocessing/preprocess_for_plm.py --clf_task tok_clf -plm roberta

# SEQUENTIAL SENTENCE CLASSIFICATION

# seq_sent_clf, len 5
python3 preprocessing/preprocess_for_plm.py --clf_task seq_sent_clf --sequence_length 5 --windowed
python3 preprocessing/preprocess_for_plm.py --clf_task seq_sent_clf --sequence_length 5
# seq_sent_clf, len 10
python3 preprocessing/preprocess_for_plm.py --clf_task seq_sent_clf --sequence_length 10 --windowed
python3 preprocessing/preprocess_for_plm.py --clf_task seq_sent_clf --sequence_length 10



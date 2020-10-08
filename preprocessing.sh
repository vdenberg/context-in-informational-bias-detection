# BASELINES

# sentence classification, BERT
python3 preprocessing/preprocessing_for_plm.py --clf_task sent_clf -plm BERT
# sentence classification, RoBERTa
python3 preprocessing/preprocessing_for_plm.py --clf_task sent_clf -plm RoBERTa

# baseline, token classification, BERT
python3 preprocessing/preprocessing_for_plm.py --clf_task tok_clf -plm BERT
# baseline, token classification, RoBERTa
python3 preprocessing/preprocessing_for_plm.py --clf_task tok_clf -plm RoBERTa

# SEQUENTIAL SENTENCE CLASSIFICATION

# seq_sent_clf, len 5
python3 preprocessing/preprocessing_for_plm.py --clf_task seq_sent_clf --sequence_length 5 --windowed
python3 preprocessing/preprocessing_for_plm.py --clf_task seq_sent_clf --sequence_length 5
# seq_sent_clf, len 10
python3 preprocessing/preprocessing_for_plm.py --clf_task seq_sent_clf --sequence_length 10 --windowed
python3 preprocessing/preprocessing_for_plm.py --clf_task seq_sent_clf --sequence_length 10

# CIM & DOMAIN

python3 preprocessing/preprocessing_for_cim_and_tapt.py # --add_use --add_sbert





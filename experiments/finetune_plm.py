from __future__ import absolute_import, division, print_function
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from lib.classifiers.PLMWrapper import Inferencer, save_model, load_features
from lib.classifiers.PLMWrapper import BertForTokenClassification, BertForSequenceClassification
from lib.classifiers.PLMWrapper import RobertaForTokenClassification, RobertaForSequenceClassification
from lib.classifiers.PLMWrapper import RobertaForSequentialSequenceClassification
from lib.handle_data.PreprocessForRoberta import *
from datetime import datetime
import torch
import random, argparse
import numpy as np
import pandas as pd
import os, sys
from lib.utils import get_torch_device
import logging

'''
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, my_id, input_ids, input_mask, segment_ids, label_id):
        self.my_id = my_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
'''


def select_model(model, clf_task):
    if clf_task == 'sent_clf':
        if model == 'bert':
            sel_mod = BertForSequenceClassification
        else:
            sel_mod = RobertaForSequenceClassification

    elif clf_task == 'tok_clf':
        if model == 'bert':
            sel_mod = BertForTokenClassification
        else:
            sel_mod = RobertaForTokenClassification

    elif clf_task == 'seq_sent_clf':
        sel_mod = RobertaForSequentialSequenceClassification
    return sel_mod


def clean_mean(df, grby='', set_type=''):
    mets = ['f1']
    if set_type:
        tmp_df = df[df.set_type == set_type]
    else:
        tmp_df = df
    return tmp_df.groupby(grby)[mets].mean().round(2)

########################
# WHAT IS THE EXPERIMENT
########################


# find GPU if present
model_mapping = {'bert': 'bert-base-cased',
                 'rob_base': 'roberta-base',
                 'rob_dapt': 'experiments/adapt_dapt_tapt/pretrained_models/news_roberta_base',
                 'rob_basil_tapt': 'experiments/adapt_dapt_tapt/dont-stop-pretraining/roberta-tapt',
                 'rob_basil_dapttapt': 'experiments/adapt_dapt_tapt/dont-stop-pretraining/roberta-dapttapt',
                }

model_hyperparams = {'sent_clf': {
                        'bert': {'lr': 2e-5, 'bs': 16, 'seeds': [6, 11, 20, 22, 34],},
                        'rob_base': {'lr': 1e-5, 'bs': 16, 'seeds': [49, 57, 33, 297, 181]},
                        'rob_dapt':  {'lr': 1e-5, 'bs': 16, 'seeds': [6, 22, 33, 34, 49]},
                        'rob_basil_tapt':  {'lr': 1e-5, 'bs': 16, 'seeds': [6, 33, 34, 49, 181]},
                        'rob_basil_dapttapt':  {'lr': 1e-5, 'bs': 16, 'seeds': [6, 33, 34, 49, 181]}
                        },
                     'tok_clf': {
                        'bert': {'lr': 2e-5, 'bs': 16, 'seeds': [6, 23, 49, 132, 281],},
                        'rob_base': {'lr': 1e-5, 'bs': 16, 'seeds': [6, 33, 34, 132, 281]}
                        },
                     'seq_sent_clf': {
                        'rob_base': {'lr': 1.5e-5, 'bs': 4, 'seeds': [22, 34, 49, 181, 43]},
                        }
                    }

device, USE_CUDA = get_torch_device()

parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--n_epochs', type=int, default=10) #2,3,4
parser.add_argument('-debug', '--debug', action='store_true', default=False)
parser.add_argument('-sampler', '--sampler', type=str, default='sequential')
parser.add_argument('-clf_task', '--clf_task', help='tok_clf|sent_clf', type=str, default='sent_clf')
parser.add_argument('-spl', '--split', type=str, default='story_split')  # sentence or story
parser.add_argument('-model', '--model', help='bert|rob_base',type=str, default='rob_base')
parser.add_argument('-lr', '--lr', type=float, default=None)
parser.add_argument('-bs', '--bs', type=int, default=None)
parser.add_argument('-sv', '--sv', type=int, default=None)
parser.add_argument('-embeds', '--embeds', action='store_true', default=False)
parser.add_argument('-win', '--window', action='store_true', default=False)
parser.add_argument('-exlen', '--example_length', help='5|10',type=int, default=5)
parser.add_argument('-force_pred', '--force_pred', action='store_true', default=False)
parser.add_argument('-force_train', '--force_train', action='store_true', default=False)
args = parser.parse_args()

N_EPS = args.n_epochs
SAMPLER = args.sampler
CLF_TASK = args.clf_task
SPLIT = args.split
MODEL = args.model
WINDOW = args.window
EX_LEN = args.example_length

task_name_elements = []
if CLF_TASK == 'seq_sent_len':
    if WINDOW:
        task_name_elements.append('w')
task_name_elements.append(CLF_TASK)
if CLF_TASK == 'seq_sent_len':
    task_name_elements.append(str(EX_LEN))
task_name_elements.extend([SPLIT, MODEL])
TASK_NAME = '_'.join(task_name_elements)

NUM_LABELS = 4 if CLF_TASK == 'seq_sent_clf' else 2

STORE_EMBEDS = args.embeds
models = [args.model]
seeds = [args.sv] if args.sv else model_hyperparams[CLF_TASK][MODEL]['seeds']
bss = [args.bs] if args.bs else [model_hyperparams[CLF_TASK][MODEL]['bs']]
lrs = [args.lr] if args.lr else [model_hyperparams[CLF_TASK][MODEL]['lr']]

if SPLIT == 'story_split':
    folds = [str(el) for el in range(1,11)]
elif MODEL != 'bert' and CLF_TASK == 'sent_clf':
    folds = ['fan']
else:
    folds = ['sentence_split']

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEBUG = args.debug
FORCE_PRED = args.force_pred
FORCE_TRAIN = args.force_train
if DEBUG:
    FORCE_PRED = True
    FORCE_TRAIN = True
    N_EPS = 2
    seeds = [0]
    if CLF_TASK != 'seq_sent_clf':
        bss = [32]
        lrs = [3e-5]
    folds = ['1']

########################
# WHERE ARE THE FILES
########################

if CLF_TASK == 'seq_sent_clf':
    if WINDOW:
        FEAT_DIR = f'data/inputs/{CLF_TASK}/windowed/ssc{EX_LEN}/'
    else:
        FEAT_DIR = f'data/inputs/{CLF_TASK}/non_windowed/ssc{EX_LEN}/'

elif CLF_TASK == 'sent_clf':
    if MODEL == 'bert':
        FEAT_DIR = f'data/inputs/{CLF_TASK}/features_for_bert'
    else:
        FEAT_DIR = f'/home/mitarb/vdberg/Projects/EntityFramingDetection/data/{CLF_TASK}/features_for_roberta'

else:
    if MODEL == 'bert':
        FEAT_DIR = f'data/inputs/{CLF_TASK}/features_for_bert'
    else:
        FEAT_DIR = f'data/inputs/{CLF_TASK}/features_for_roberta'

PREDICTION_DIR = f'reports/{CLF_TASK}/{TASK_NAME}/tables'
CHECKPOINT_DIR = f'models/checkpoints/{TASK_NAME}'
if MODEL != 'bert' and CLF_TASK == 'sent_clf':
    CHECKPOINT_DIR = f'/home/mitarb/vdberg/Projects/EntityFramingDetection/models/checkpoints/SC_rob/'
REPORTS_DIR = f'reports/{CLF_TASK}/{TASK_NAME}/logs'
TABLE_DIR = f'reports/{CLF_TASK}/{TASK_NAME}/tables'
CACHE_DIR = 'models/cache/'
MAIN_TABLE_FP = os.path.join(TABLE_DIR, f'{TASK_NAME}_results.csv')

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
if not os.path.exists(TABLE_DIR):
    os.makedirs(TABLE_DIR)

table_columns = 'model,sampler,seed,bs,lr,model_loc,fold,epochs,set_type,loss,fn,fp,tn,tp,acc,prec,rec,f1'
main_results_table = pd.DataFrame(columns=table_columns.split(','))

########################
# MAIN
########################

PRINT_EVERY = 100
logger = logging.getLogger()
inferencer = Inferencer(REPORTS_DIR, logger, device, use_cuda=USE_CUDA)

if __name__ == '__main__':

    # set logger
    now = datetime.now()
    now_string = now.strftime(format=f'%b-%d-%Hh-%-M_{TASK_NAME}')
    LOG_FP = f"{REPORTS_DIR}/{now_string}.log"
    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=LOG_FP)
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
    logger = logging.getLogger()
    logger.info(args)

    for MODEL in models:
        EXACT_MODEL = model_mapping[MODEL]

        for SEED in seeds:
            if SEED == 0:
                SEED_VAL = random.randint(0, 300)
            else:
                SEED_VAL = SEED

            seed_name = f"{MODEL}_{SAMPLER}_{SEED_VAL}"
            random.seed(SEED_VAL)
            np.random.seed(SEED_VAL)
            torch.manual_seed(SEED_VAL)
            torch.cuda.manual_seed_all(SEED_VAL)

            prediction_dir = f'data/predictions/{TASK_NAME}/'
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir)

            embedding_dir = f'/home/mitarb/vdberg/Projects/EntityFramingDetection/data/embeddings/{MODEL}/'
            if not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir)

            for BATCH_SIZE in bss:
                bs_name = seed_name + f"_bs{BATCH_SIZE}"
                for LEARNING_RATE in lrs:
                    setting_name = bs_name + f"_lr{LEARNING_RATE}"
                    setting_results_table = pd.DataFrame(columns=table_columns.split(','))

                    pred_fp = os.path.join(prediction_dir, f'{setting_name}_test_preds.csv')

                    test_ids = []
                    test_predictions = []
                    test_labels = []
                    test_res = {'model': MODEL, 'seed': SEED_VAL, 'fold': SPLIT, 'bs': BATCH_SIZE,
                                 'lr': LEARNING_RATE, 'set_type': 'test', 'sampler': SAMPLER}

                    feat_fp = os.path.join(FEAT_DIR, f"all_features.pkl")
                    all_ids, all_batches, all_labels = load_features(feat_fp, batch_size=1,
                                                                     sampler=SAMPLER)

                    for fold_name in folds:
                        fold_results_table = pd.DataFrame(columns=table_columns.split(','))
                        name = setting_name + f"_f{fold_name}"

                        # init results containers
                        if MODEL != 'bert' and fold_name == 'sentence_split' and CLF_TASK == 'sent_clf':
                            model_loc_name = setting_name + f"_f{'fan'}"
                        else:
                            model_loc_name = name
                        best_model_loc = os.path.join(CHECKPOINT_DIR, model_loc_name)
                        best_val_res = {'model': MODEL, 'seed': SEED_VAL, 'fold': fold_name, 'bs': BATCH_SIZE,
                                        'lr': LEARNING_RATE, 'set_type': 'dev', 'f1': 0, 'model_loc': best_model_loc,
                                        'sampler': SAMPLER, 'epochs': N_EPS}

                        # load feats
                        train_fp = os.path.join(FEAT_DIR, f"{fold_name}_train_features.pkl")
                        dev_fp = os.path.join(FEAT_DIR, f"{fold_name}_dev_features.pkl")
                        test_fp = os.path.join(FEAT_DIR, f"{fold_name}_test_features.pkl")
                        _, train_batches, train_labels = load_features(train_fp, BATCH_SIZE, SAMPLER)
                        _, dev_batches, dev_labels = load_features(dev_fp, 1, SAMPLER)
                        fold_test_ids, test_batches, fold_test_labels = load_features(test_fp, 1, SAMPLER)
                        test_ids.extend(fold_test_ids)
                        test_labels.extend(fold_test_labels)

                        if not os.path.exists(pred_fp) or FORCE_PRED:

                            # start training
                            logger.info(f"***** Init {CLF_TASK} {fold_name} *****")
                            logger.info(f"  Details: {best_val_res}")
                            logger.info(f"  Logging to {LOG_FP}")

                            selected_model = select_model(MODEL, CLF_TASK)

                            if not os.path.exists(best_model_loc) or FORCE_TRAIN:

                                model = selected_model.from_pretrained(EXACT_MODEL, cache_dir=CACHE_DIR,
                                                                       num_labels=NUM_LABELS,
                                                                       output_hidden_states=True,
                                                                       output_attentions=False)

                                model.to(device)

                                BERT_TOK_CLF = (CLF_TASK == 'tok_clf') & (MODEL == 'bert')
                                if not BERT_TOK_CLF:
                                    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01, eps=1e-6)
                                    n_train_batches = len(train_batches)
                                    half_train_batches = int(n_train_batches / 2)
                                    GRADIENT_ACCUMULATION_STEPS = 2
                                    WARMUP_PROPORTION = 0.06
                                    num_tr_opt_steps = n_train_batches * N_EPS / GRADIENT_ACCUMULATION_STEPS
                                    num_tr_warmup_steps = int(WARMUP_PROPORTION * num_tr_opt_steps)
                                    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                                                num_warmup_steps=num_tr_warmup_steps,
                                                                                num_training_steps=num_tr_opt_steps)
                                else:
                                    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
                                    scheduler = None

                                model.train()

                                logger.info(f"***** Train {CLF_TASK} {name} *****")
                                logger.info(f"  Details: {best_val_res}")
                                logger.info(f"  Logging to {LOG_FP}")

                                for ep in range(1, N_EPS + 1):
                                    epoch_name = name + f"_ep{ep}"
                                    tr_loss = 0
                                    for step, batch in enumerate(train_batches):
                                        batch = tuple(t.to(device) for t in batch)

                                        model.zero_grad()
                                        outputs = model(batch[0], batch[1], labels=batch[2])
                                        loss = outputs[0]

                                        print(batch[0], batch[1], batch[2])
                                        print(loss)
                                        exit(0)
                                        loss.backward()
                                        tr_loss += loss.item()
                                        optimizer.step()
                                        if scheduler:
                                            scheduler.step()

                                        if step % PRINT_EVERY == 0 and step != 0:
                                            logging.info(f' Ep {ep} / {N_EPS} - {step} / {len(train_batches)} - Loss: {loss.item()}')

                                    av_loss = tr_loss / len(train_batches)

                                    dev_mets, dev_perf = inferencer.evaluate(model, dev_batches, dev_labels,
                                                                                av_loss=av_loss, set_type='dev',
                                                                                name=epoch_name, output_mode=CLF_TASK)

                                    # check if best
                                    high_score = ''
                                    if dev_mets['f1'] > best_val_res['f1']:
                                        best_val_res.update(dev_mets)
                                        high_score = '(HIGH SCORE)'
                                        save_model(model, CHECKPOINT_DIR, name)

                                    logger.info(f'{epoch_name}: {dev_perf} {high_score}')

                            best_model = selected_model.from_pretrained(best_model_loc, num_labels=NUM_LABELS,
                                                                        output_hidden_states=True,
                                                                        output_attentions=False)
                            best_model.to(device)

                            # get predictions
                            fold_test_predictions = inferencer.predict(best_model, test_batches, output_mode=CLF_TASK)
                            test_predictions.extend(fold_test_predictions)

                            # get embeddings
                            if STORE_EMBEDS:
                                for EMB_TYPE in ['cross4bert']: #poolbert', 'avbert', 'unpoolbert', 'crossbert'
                                    emb_fp = os.path.join(embedding_dir, f'{name}_basil_w_{EMB_TYPE}')

                                    PREFERRED_EMB_SV = 49
                                    if SEED_VAL == PREFERRED_EMB_SV and not os.path.exists(emb_fp):
                                        logging.info(f'Generating {EMB_TYPE} ({emb_fp})')
                                        embs = inferencer.predict(best_model, all_batches, return_embeddings=True, emb_type=EMB_TYPE)
                                        assert len(embs) == len(all_ids)

                                        basil_w_BERT = pd.DataFrame(index=all_ids)
                                        basil_w_BERT[EMB_TYPE] = embs
                                        basil_w_BERT.to_csv(emb_fp)
                                        logger.info(f'{EMB_TYPE} embeddings in {emb_fp}.csv')

                            # store performance on just the fold in the table
                            fold_results_table = fold_results_table.append(best_val_res, ignore_index=True)

                    logger.info(f"***** Predict {CLF_TASK} {name} *****")
                    logger.info(f"  Details: {best_val_res}")
                    logger.info(f"  Logging to {LOG_FP}")

                    if not os.path.exists(pred_fp) or FORCE_PRED:
                        # compute performance on setting
                        assert len(test_predictions) == len(test_ids)
                        assert len(test_predictions) == len(test_labels)
                        basil_w_pred = pd.DataFrame(index=test_ids)
                        basil_w_pred['pred'] = test_predictions
                        basil_w_pred['label'] = test_labels
                        basil_w_pred.to_csv(pred_fp)

                    # load predictions
                    basil_w_pred = pd.read_csv(pred_fp) #, dtype={'pred': np.int64})
                    logger.info(f'Preds from {pred_fp}')

                    logger.info(f"***** Eval {CLF_TASK} {name} *****")
                    logger.info(f"  Details: {best_val_res}")
                    logger.info(f"  Logging to {LOG_FP}")

                    test_mets, test_perf = inferencer.evaluate(labels=basil_w_pred.label, preds=basil_w_pred.pred,
                                                               set_type='test', name=setting_name,
                                                               output_mode=CLF_TASK)

                    logging.info(f"{test_perf}")
                    test_res.update(test_mets)

                    setting_results_table = setting_results_table.append(test_res, ignore_index=True)
                    setting_results_table = setting_results_table.append(fold_results_table, ignore_index=True)
                    setting_fp = os.path.join(TABLE_DIR, f'{setting_name}_results_table.csv')

                    #if os.path.exists(setting_fp):
                    #    orig_setting_results_table = pd.read_csv(setting_fp)
                    #    setting_results_table = pd.concat((orig_setting_results_table, setting_results_table))
                    #    setting_results_table = setting_results_table.drop_duplicates(keep='first')

                    setting_results_table.to_csv(setting_fp, index=False)

                    # store performance of setting
                    main_results_table = main_results_table.append(setting_results_table, ignore_index=True)

        main_results_table.to_csv(MAIN_TABLE_FP, index=False)

        df = main_results_table
        df[['prec', 'rec', 'f1']] = df[['prec', 'rec', 'f1']].round(4) * 100
        df = df.fillna(0)
        print(df[['model', 'seed', 'set_type', 'seed', 'prec', 'rec', 'f1']])

        view = clean_mean(df, grby=['model', 'seed'], set_type='test')
        view = view.fillna(0)
        print(view)

        test = df[df.set_type == 'test']
        test = test[['set_type', 'seed', 'prec', 'rec', 'f1']]
        test = test.groupby('seed').mean()
        test = test.describe()
        test_m = test.loc['mean'].round(2).astype(str)
        test_std = test.loc['std'].round(2).astype(str)
        result = test_m + ' \pm ' + test_std
        print(f"\n{TASK_NAME} results on {SPLIT}:")
        print(main_results_table.seed.unique())
        print(result)

        logger.info(f"  Log in {LOG_FP}")
        logger.info(f"  Table in {MAIN_TABLE_FP}")
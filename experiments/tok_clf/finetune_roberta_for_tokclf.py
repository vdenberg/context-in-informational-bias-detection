from __future__ import absolute_import, division, print_function

import random, argparse
from datetime import datetime
import numpy as np
import pandas as pd

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch

from lib.handle_data.PreprocessForPLM import *
from lib.utils import get_torch_device

from lib.classifiers.RobertaWrapper import RobertaForTokenClassification, Inferencer, save_model, load_features


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, my_id, input_ids, input_mask, segment_ids, label_id):
        self.my_id = my_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


########################
# WHAT IS THE EXPERIMENT
########################

model_mapping = {'rob_base': 'roberta-base'}
device, USE_CUDA = get_torch_device()

parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--n_epochs', type=int, default=10) #2,3,4
parser.add_argument('-sampler', '--sampler', type=str, default='sequential')
parser.add_argument('-model', '--model', type=str, default=None) #2,3,4
parser.add_argument('-lr', '--lr', type=float, default=None) #5e-5, 3e-5, 2e-5
parser.add_argument('-bs', '--bs', type=int, default=None) #16, 21
parser.add_argument('-sv', '--sv', type=int, default=None) #16, 21
parser.add_argument('-fold', '--fold', type=str, default=None) #16, 21
args = parser.parse_args()

N_EPS = args.n_epochs
models = [args.model] if args.model else ['rob_base']
seeds = [args.sv] if args.sv else [33, 22, 6, 181, 49]
bss = [args.bs] if args.bs else [16]
lrs = [args.lr] if args.lr else [1e-5]
folds = [args.fold] if args.fold else [str(el+1) for el in range(10)] + ['fan']
SAMPLER = 'sequential'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEBUG = args.debug
if DEBUG:
    N_EPS = 2
    seeds = [0]
    bss = [32]
    lrs = [3e-5]
    folds = ['1']

########################
# WHERE ARE THE FILES
########################

TASK_NAME = f'tok_clf_roberta'
TASK = 'tok_clf'
FEAT_DIR = f'data/inputs/tok_clf/features_for_roberta'
CHECKPOINT_DIR = f'models/checkpoints/{TASK_NAME}/'
REPORTS_DIR = f'reports/{TASK}/{TASK_NAME}/logs'
TABLE_DIR = f'reports/{TASK}/{TASK_NAME}/tables'
CACHE_DIR = 'models/cache/'
MAIN_TABLE_FP = os.path.join(TABLE_DIR, f'roberta_ft_results.csv')

if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
if not os.path.exists(TABLE_DIR):
    os.makedirs(TABLE_DIR)
if os.path.exists(MAIN_TABLE_FP):
    table_columns = 'model,sampler,seed,bs,lr,model_loc,fold,epoch,set_type,rep_sim,loss,fn,fp,tn,tp,acc,prec,rec,f1'
    main_results_table = pd.read_csv(MAIN_TABLE_FP)
else:
    table_columns = 'model,sampler,seed,bs,lr,model_loc,fold,epoch,set_type,rep_sim,loss,fn,fp,tn,tp,acc,prec,rec,f1'
    main_results_table = pd.DataFrame(columns=table_columns.split(','))

########################
# MAIN
########################

NUM_LABELS = 4
PRINT_EVERY = 100

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
        ROBERTA_MODEL = model_mapping[MODEL]

        for SEED in seeds:
            if SEED == 0:
                SEED_VAL = random.randint(0, 300)
            else:
                SEED_VAL = SEED

            seed_name = f"{MODEL}_tok_{SEED_VAL}"
            random.seed(SEED_VAL)
            np.random.seed(SEED_VAL)
            torch.manual_seed(SEED_VAL)
            torch.cuda.manual_seed_all(SEED_VAL)

            for BATCH_SIZE in bss:
                bs_name = seed_name + f"_bs{BATCH_SIZE}"
                for LEARNING_RATE in lrs:
                    setting_name = bs_name + f"_lr{LEARNING_RATE}"
                    setting_results_table = pd.DataFrame(columns=table_columns.split(','))
                    for fold_name in folds:
                        fold_results_table = pd.DataFrame(columns=table_columns.split(','))
                        name = setting_name + f"_f{fold_name}"

                        # init results containers
                        best_model_loc = os.path.join(CHECKPOINT_DIR, name)
                        best_val_res = {'model': MODEL, 'seed': SEED_VAL, 'fold': fold_name, 'bs': BATCH_SIZE, 'lr': LEARNING_RATE, 'set_type': 'dev',
                                        'f1': 0, 'model_loc': best_model_loc, 'sampler': SAMPLER}
                        test_res = {'model': MODEL, 'seed': SEED_VAL, 'fold': fold_name, 'bs': BATCH_SIZE, 'lr': LEARNING_RATE, 'set_type': 'test',
                                    'sampler': SAMPLER}

                        # load feats
                        train_fp = os.path.join(FEAT_DIR, f"{fold_name}_train_features.pkl")
                        dev_fp = os.path.join(FEAT_DIR, f"{fold_name}_dev_features.pkl")
                        test_fp = os.path.join(FEAT_DIR, f"{fold_name}_test_features.pkl")
                        train_ids, train_batches, train_labels = load_features(train_fp, BATCH_SIZE, SAMPLER)
                        dev_ids, dev_batches, dev_labels = load_features(dev_fp, BATCH_SIZE, SAMPLER)
                        test_ids, test_batches, test_labels = load_features(test_fp, BATCH_SIZE, SAMPLER)

                        # start training
                        logger.info(f"***** Training on Fold {fold_name} *****")
                        logger.info(f"  Details: {best_val_res}")
                        logger.info(f"  Logging to {LOG_NAME}")

                        FORCE = True
                        if not os.path.exists(best_model_loc) or FORCE:
                            model = RobertaForTokenClassification.from_pretrained(ROBERTA_MODEL,
                                                                                     cache_dir=CACHE_DIR,
                                                                                     num_labels=NUM_LABELS,
                                                                                     output_hidden_states=True,
                                                                                     output_attentions=False)
                            model.to(device)
                            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01,
                                              eps=1e-6)  # To reproduce BertAdam specific behavior set correct_bias=False

                            n_train_batches = len(train_batches)
                            half_train_batches = int(n_train_batches / 2)
                            GRADIENT_ACCUMULATION_STEPS = 2
                            WARMUP_PROPORTION = 0.06
                            num_tr_opt_steps = n_train_batches * N_EPS / GRADIENT_ACCUMULATION_STEPS
                            num_tr_warmup_steps = int(WARMUP_PROPORTION * num_tr_opt_steps)
                            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                                        num_warmup_steps=num_tr_warmup_steps,
                                                                        num_training_steps=num_tr_opt_steps)

                            model.train()

                            for ep in range(1, N_EPS + 1):
                                epoch_name = name + f"_ep{ep}"
                                tr_loss = 0
                                for step, batch in enumerate(train_batches):
                                    batch = tuple(t.to(device) for t in batch)
                                    print(batch[-1])
                                    model.zero_grad()
                                    outputs = model(batch[0], batch[1], labels=batch[-1])

                                    loss = outputs[0]

                                    loss.backward()
                                    tr_loss += loss.item()
                                    optimizer.step()
                                    scheduler.step()

                                    if step % PRINT_EVERY == 0 and step != 0:
                                        logging.info(f' Ep {ep} / {N_EPS} - {step} / {len(train_batches)} - Loss: {loss.item()}')

                                av_loss = tr_loss / len(train_batches)
                                # save_model(model, CHECKPOINT_DIR, epoch_name)
                                dev_mets, dev_perf = inferencer.evaluate(model, dev_batches, dev_labels, av_loss=av_loss,
                                                                         set_type='dev', name=epoch_name, output_mode=TASK)

                                # check if best
                                high_score = ''
                                if dev_mets['f1'] > best_val_res['f1']:
                                    best_val_res.update(dev_mets)
                                    high_score = '(HIGH SCORE)'
                                    save_model(model, CHECKPOINT_DIR, name)

                                logger.info(f'{epoch_name}: {dev_perf} {high_score}')

                        best_model = RobertaForTokenClassification.from_pretrained(best_model_loc,
                                                                                      num_labels=NUM_LABELS,
                                                                                      output_hidden_states=True,
                                                                                      output_attentions=False)
                        best_model.to(device)

                        logger.info(f"***** Best model on Fold {fold_name} *****")
                        logger.info(f"  Details: {best_val_res}")

                        dev_mets, dev_perf = inferencer.evaluate(best_model, dev_batches, dev_labels, set_type='dev', output_mode=TASK)
                        best_val_res.update(dev_mets)
                        logging.info(f"{dev_perf}")

                        test_mets, test_perf = inferencer.evaluate(best_model, test_batches, test_labels, set_type='test', output_mode=TASK)
                        test_res.update(test_mets)
                        logging.info(f"{test_perf}")

                        # store performance in table
                        fold_results_table = fold_results_table.append(best_val_res, ignore_index=True)
                        fold_results_table = fold_results_table.append(test_res, ignore_index=True)
                        setting_results_table = setting_results_table.append(fold_results_table)

                    # print result of setting
                    logging.info(
                        f'Setting {setting_name} results: \n{setting_results_table[["model", "seed", "bs", "lr", "fold", "set_type", "f1"]]}')

                    # store performance of setting
                    main_results_table = main_results_table.append(setting_results_table, ignore_index=True)

                    # write performance to file
                    setting_fp = os.path.join(TABLE_DIR, f'{setting_name}_results_table.csv')

                    if os.path.exists(setting_fp):
                        orig_setting_results_table = pd.read_csv(setting_fp)
                        setting_results_table = pd.concat((orig_setting_results_table, setting_results_table))
                        setting_results_table = setting_results_table.drop_duplicates(keep='first')

                    setting_results_table.to_csv(setting_fp, index=False)

                    logger.info(f"  Log in {LOG_FP}")
                    logger.info(f"  Table in {setting_fp}")

                    main_results_table.to_csv(MAIN_TABLE_FP, index=False)


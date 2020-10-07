import time, os
from lib.utils import format_runtime, plot_scores
from lib.evaluate.Eval import my_eval

from sklearn.model_selection import learning_curve
import torch

class Classifier:
    """
    Generic Classifier that performs recurring machine learning tasks
    """
    def __init__(self, model, logger, fig_dir, name, n_eps=10, patience=3, printing=100, load_from_ep=None):
        self.wrapper = model
        self.n_epochs = n_eps
        self.logger = logger
        self.patience = patience
        self.fig_dir = fig_dir
        self.model_name = name
        self.print_every = printing

        # load
        self.epochs = range(n_eps)
        if load_from_ep:
            self.n_epochs += load_from_ep
            self.epochs = range(load_from_ep, self.n_epochs)
        else:
            self.epochs = range(1, self.n_epochs+1)

        # empty now and set during or after training
        self.train_time = 0
        self.prev_val_f1 = 0
        self.best_val_mets = {'f1':0}
        self.best_val_perf = ''
        self.full_patience = patience
        self.current_patience = self.full_patience
        self.test_mets = {}
        self.test_perf_string = ''
        self.cur_fold = ''
        self.best_model_loc = ''

    def train_epoch(self, train_batches):
        start = time.time()
        epoch_loss = 0
        for step, batch in enumerate(train_batches):

            loss = self.wrapper.train_on_batch(batch)
            epoch_loss += loss

            if (step > 0) & (step % self.print_every == 0):
                self.logger.info(f' > Step {step}/{len(train_batches)}: loss = {round(epoch_loss/step,4)}')

        av_epoch_loss = epoch_loss / len(train_batches)
        elapsed = format_runtime(time.time() - start)
        return av_epoch_loss, elapsed

    def update_patience(self, val_f1):
        # if an improvement happens, we have full patience, if no improvement happens
        # patience goes down, if patience reaches zero, we stop training
        if val_f1 > self.prev_val_f1:
            self.current_patience = self.full_patience
        else:
            self.current_patience -= 1
        self.prev_val_f1 = val_f1

    def unpack_fold(self, fold, voter_i):
        self.cur_fold = fold['name']
        tr_bs, tr_lbs = fold['train_batches'][voter_i], fold['train'][voter_i].label
        dev_bs, dev_lbs = fold['dev_batches'][voter_i], fold['dev'][voter_i].label
        return tr_bs, tr_lbs, dev_bs, dev_lbs

    def validate_after_epoch(self, ep, elapsed, fold, voter_i):
        ep_name = self.model_name + f"_ep{ep}"

        tr_bs, tr_lbs, dev_bs, dev_lbs = self.unpack_fold(fold, voter_i=voter_i)

        tr_preds, tr_loss, _, _ = self.wrapper.predict(tr_bs)
        tr_mets, tr_perf = my_eval(tr_lbs, tr_preds, set_type='train', av_loss=tr_loss, name="")

        val_preds, val_loss, _, _ = self.wrapper.predict(dev_bs)
        val_mets, val_perf = my_eval(dev_lbs, val_preds, set_type='dev', av_loss=val_loss, name="")

        best_log = ''
        if val_mets['f1'] > self.best_val_mets['f1']:
            self.best_val_mets = val_mets
            self.best_val_mets['epoch'] = ep
            self.best_model_loc = ep_name
            self.wrapper.save_model(self.model_name)
            best_log = '(HIGH SCORE)'

        test_mets, test_perf = self.test_model(fold, '')

        self.logger.info(f" Ep {ep} ({self.model_name.replace('_', '')}): "
                         f"{tr_perf} | {val_perf} {best_log}")
        self.logger.info(f" {test_perf}")
        return tr_mets, tr_perf, val_mets, val_perf

    def train_all_epochs(self, fold, voter_i):
        tr_bs, tr_lbs, dev_bs, dev_lbs = self.unpack_fold(fold, voter_i)
        train_start = time.time()
        losses = []

        if self.model_name == 'BERT':
            elapsed = format_runtime(time.time() - train_start)
            tr_mets, tr_perf, val_mets, val_perf = self.validate_after_epoch(-1, elapsed, fold, voter_i)
            losses.append((tr_mets['loss'], val_mets['loss']))

        for ep in self.epochs:
            self.wrapper.model.train()

            av_tr_loss, ep_elapsed = self.train_epoch(tr_bs)

            tr_mets, tr_perf, val_mets, val_perf = self.validate_after_epoch(ep, ep_elapsed, fold, voter_i)
            losses.append((av_tr_loss, val_mets['loss']))

            self.update_patience(val_mets['f1'])

            if (not self.current_patience > 0) & (val_mets['f1'] > 0.20):
                self.logger.info(" > Stopping training.")
                break

        eps_elapsed = format_runtime(time.time() - train_start)
        return eps_elapsed, losses

    def test_model(self, fold, name):
        preds, test_loss, _, _ = self.wrapper.predict(fold['test_batches'])
        test_mets, test_perf = my_eval(fold['test'].label, preds, name=name, set_type='test', av_loss=test_loss)
        return test_mets, test_perf

    def train_on_fold(self, fold, voter_i):
        self.cur_fold = fold['name']
        train_elapsed, losses = self.train_all_epochs(fold, voter_i)
        self.train_time = train_elapsed

        # plot learning curve
        #loss_plt = plot_scores(losses)
        #loss_plt.savefig(self.fig_dir + f'/{self.model_name}_trainval_loss.png', bbox_inches='tight')

        # test_model
        if self.best_model_loc:
            self.wrapper.load_model(self.model_name)
            self.logger.info(f'Loaded best model from {self.best_model_loc}')

            name = self.model_name + f"_TEST_{self.n_epochs}"

            # test_mets, test_perf = self.test_model(fold, name)
            preds, test_loss, _, _ = self.wrapper.predict(fold['test_batches'])
            test_mets, test_perf = my_eval(fold['test'].label, preds, name=name, set_type='test', av_loss=test_loss)

            self.logger.info(f' FINISHED training {name} (took {self.train_time})')
            self.logger.info(f" {test_mets}")
        else:
            test_mets = None

        return self.best_val_mets, test_mets, preds

    def produce_preds(self, fold, model_name):
        if not model_name:
            model_name = self.model_name
        self.wrapper.load_model(model_name)
        preds, _, _, losses = self.wrapper.predict(fold['test_batches'])
        return preds, losses






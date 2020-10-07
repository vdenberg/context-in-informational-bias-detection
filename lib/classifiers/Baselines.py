# !pip install ktrain
# !pip install --upgrade tensorflow==2.0

import ktrain
from ktrain import text
from lib.handle_data.SplitData import Split
from lib.evaluate.StandardEval import evaluation

import pandas
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy


# =====================================================================================
#                    TFIDF
# =====================================================================================

def get_tfidf_mean(tfidf, sent):
    scores = tfidf.transform([sent])
    for row in scores:
        scores = [cell for cell in row.data if cell > 0]
        scores = scores if scores else [1]
    average = sum(scores) / len(scores)
    return average


class TFIDFBaseline:
    def __init__(self, basil):
        self.basil = basil
        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(basil.tokens)

    def rank_sentences(self):
        low_tfidf = pandas.DataFrame(numpy.zeros(self.basil.bias.shape), index=self.basil.index, columns=['low_tfidf'])
        for n, story in self.basil.groupby('story'):
            for n2, article in story.groupby('source'):
                av_scores = article.tokens.apply(lambda x: get_tfidf_mean(self.tfidf, x))
                picks = av_scores.nsmallest(4)
                low_tfidf.loc[picks.index, 'low_tfidf'] = [1] * 4
        return low_tfidf

    def get_y_pred(self, train_X, train_y, test_X, test_y):
        y_pred = test_X.low_tfidf
        return train_X, train_y, test_X, test_y, y_pred

# =====================================================================================
#                    MAJORITY
# =====================================================================================

class Maj:
    def __init__(self):
        self.maj = None

    def train(self, X, y):
        self.maj = y.mode().values[0]

    def predict(self, X):
        return [self.maj] * len(X)

    def get_y_pred(self, train_X, train_y, test_X, test_y):
        self.train(train_X, train_y)
        y_pred = self.predict(test_X)
        return train_X, train_y, test_X, test_y, y_pred

# =====================================================================================
#                    Finetuned BERT
# =====================================================================================


# test if GPU can be found
import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')


# load and split data
BASIL = 'data/basil.csv' #'/content/drive/My Drive/InformationalBiasDetection/basil.csv'
SPL = 'data/split/split.json' #/content/drive/My Drive/InformationalBiasDetection/split.json'
PRED = 'classifiers/models' #predictors
LS = 'classifiers/losses' #losses

basil = pd.read_csv(BASIL, index_col=0).fillna('')
spl = Split(basil, which='berg', split_fp=SPL)
folds = spl.apply_split(features=['sentence'], output_as='df')

t = text.Transformer('bert-base-uncased', maxlen=512, classes=[0, 1])
model = t.get_classifier()

for fold_i, fold in enumerate(folds):
    ids_train, x_train, y_train = fold['train'].index.values, fold['train']['sentence'].to_list(), fold['train']['bias'].values
    ids_dev, x_dev, y_dev = fold['dev'].index.values, fold['dev']['sentence'].to_list(), fold['dev']['bias'].values
    ids_test, x_test, y_test = fold['test'].index.values, fold['dev']['sentence'].to_list(), fold['dev']['bias'].values

    print(fold_i, 'preprocess')
    trn = t.preprocess_train(x_train, y_train)
    val = t.preprocess_test(x_dev, y_dev)

    print(fold_i, 'learn')
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)
    learner.fit_onecycle(2e-5, 4)
    learner.validate(class_names=t.get_classes())

    print(fold_i, 'analyze losses')
    lss = learner.view_top_losses(n=len(x_dev), preproc=t)
    with open(LS + '/fold{}.tsv'.format(fold_i), 'w') as f:
        for idx, loss in lss:
            f.write('{}\t{}\t{}\n'.format(ids_dev[idx], x_dev[idx], loss))

    pred_model = learner.model

    print(fold_i, 'save predictor')
    predictor = ktrain.get_predictor(pred_model, preproc=t)
    y_pred = predictor.predict(x_test)
    evalb, evalstr = evaluation(y_pred=y_pred, y_true=y_test)
    print(fold_i, evalstr, end='\n\n')

    predictor.save(PRED + '/fold{}'.format(fold_i))

# reloaded_predictor = ktrain.load_predictor('predictors/fold{}'.format(fold_i))
# y_pred = reloaded_predictor.predict(x_test)

# predictor.explain('Jesus Christ is the central figure of Christianity.')

for fold_i in range(1):
    ids_train, x_train, y_train = fold['train'].index.values, fold['train']['sentence'].to_list(), fold['train'][
        'bias'].values
    ids_dev, x_dev, y_dev = fold['dev'].index.values, fold['dev']['sentence'].to_list(), fold['dev']['bias'].values
    ids_test, x_test, y_test = fold['test'].index.values, fold['dev']['sentence'].to_list(), fold['dev']['bias'].values

    predictor = ktrain.load_predictor(PRED + ' / fold{}'.format(fold_i))

    y_pred = predictor.predict(x_dev)

    # evaluate
    y_pred = [int(el) for el in y_pred]
    y_dev = [int(el) for el in y_dev]
    evalb, evalstr = evaluation(y_pred=y_pred, y_true=y_dev)
    print(fold_i, evalstr, end='\n\n')

    compare = pd.DataFrame([y_pred, y_dev], index=ids_dev, columns=['y_pred', 'y_dev'])
    difference = compare[compare['y_pred'] != compare['y_dev']]
    print(difference)

    # predictor.explain('Jesus Christ is the central figure of Christianity.')

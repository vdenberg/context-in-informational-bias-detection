import pandas as pd
from lib.handle_data.BasilLoader import LoadBasil
import json, os

'''Proprocessing for dont-stop-pretraining scripts'''


def preprocess_basil_for_dsp(basil, test_size, train_ofp, test_ofp):
    """
    Split for tapt
    """

    basil.sample(frac=1)

    msk = np.random.rand(len(basil)) < 0.8

    if article_counter <= test_size:
        file_path = train_ofp
    else:
        file_path = test_ofp

    if file_path:
        with open(file_path, 'a') as f:
            sentences = gr.sentence.values
            for s in sentences:
                f.write(s)
                f.write('\n')
            f.write('\n')



if __name__ == '__main__':
    odir = 'experiments/dont-stop-pretraining/data'

    if not os.path.exists(odir):
        os.makedirs(odir)

    basil = LoadBasil().load_basil_raw()
    basil.to_csv('data/basil.csv')
    basil = pd.read_csv('data/basil.csv', index_col=0).fillna('')

    # Split for tapt
    train_ofp = os.path.join(odir, 'basil_train.txt')
    test_ofp = os.path.join(odir, 'basil_test.txt')
    preprocess_basil_for_dsp(basil, 250, train_ofp, test_ofp)


import pandas as pd


def load_results_file(fp):
    df = pd.read_csv(fp, index_col=False)
    df = df.fillna(0)
    metrics = ['acc', 'prec', 'rec', 'f1']
    df[metrics] = df[metrics].round(4) * 100
    return df


def clean_mean(df, grby='', set_type=''):
    mets = ['f1']
    if set_type:
        tmp_df = df[df.set_type == set_type]
    else:
        tmp_df = df
    return tmp_df.groupby(grby)[mets].mean().round(2)
import pandas as pd
from lib.ErrorAnalysis import ErrorAnalysis
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 600)
pd.set_option('display.width', 200)

ea = ErrorAnalysis('base_best')

quote_dfs = [ea.compare_subsets(ea.w_preds, 'quote', model, context) for model, context in ea.models]
quote_df = ea.concat_comparisons(quote_dfs)
print(quote_df)

df = ea.w_preds
for model, context in [('rob', '22'), ('cim', 'coverage')]: # , ('rob', '22'),  ('cim+', 'story'), ('cim++', 'story'),
    print()
    print(model, context)

    rate_of_quotes = sum(df.quote) /len(df)
    print(ea.N, rate_of_quotes)

    df_w_conf_mat = ea.conf_mat(df, model, context)

    for el in ['tp', 'fp', 'tn', 'fn']:
        subdf = df[df[el]]
        print(el)
        prop = sum(subdf.quote) / len(subdf)
        print(el, len(subdf), '\%' + str(round(prop*100,2)))

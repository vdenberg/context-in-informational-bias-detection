import pandas as pd
from lib.ErrorAnalysis import ErrorAnalysis
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 2000)

ea = ErrorAnalysis('base_best')

# LEXICAL BIAS

stance_dfs = [ea.compare_subsets(ea.w_preds, 'lex_bias', model, context) for model, context in ea.models]
stance_df = ea.concat_comparisons(stance_dfs)
print(stance_df.to_latex())


exit(0)

# SENTENCE LENGTH ANALYSIS

sentlen_dfs = [ea.compare_subsets(ea.w_preds, 'len', model, context) for model, context in ea.models]
sentlen_df = ea.concat_comparisons(sentlen_dfs)
sentlen_df.index = ["0-90", "91-137", "138-192", "193-647", "All"]
print(sentlen_df.to_latex())

# IN QUOTE OR NOT + PRESENCE OF QUOTATION MARKS

quote_dfs = [ea.compare_subsets(ea.w_preds, 'quote', model, context) for model, context in ea.models]
quote_df = ea.concat_comparisons(quote_dfs)
print(quote_df)

df_for_quote_marks = ea.w_preds
for model, context in [('rob', '22'), ('cim', 'coverage')]: # , ('rob', '22'),  ('cim+', 'story'), ('cim++', 'story'),
    print()
    print(model, context)

    rate_of_quotes = sum(df_for_quote_marks.quote) / len(df_for_quote_marks)
    print(ea.N, rate_of_quotes)

    df_w_conf_mat = ea.conf_mat(df_for_quote_marks, model, context)

    for el in ['tp', 'fp', 'tn', 'fn']:
        subdf = df_for_quote_marks[df_for_quote_marks[el]]
        print(el)
        prop = sum(subdf.quote) / len(subdf)
        print(el, len(subdf), '\%' + str(round(prop*100,2)))

# SOURCE AND STANCE

source_dfs = [ea.compare_subsets(ea.w_preds, 'source', model, context) for model, context in ea.models]
source_df = ea.concat_comparisons(source_dfs)
print(source_df.to_latex())

stance_dfs = [ea.compare_subsets(ea.w_preds, 'stance', model, context) for model, context in ea.models]
stance_df = ea.concat_comparisons(stance_dfs)
print(stance_df.to_latex())

# LEXICAL BIAS

stance_dfs = [ea.compare_subsets(ea.w_preds, 'lex_bias', model, context) for model, context in ea.models]
stance_df = ea.concat_comparisons(stance_dfs)
print(stance_df.to_latex())

import pandas as pd
from lib.ErrorAnalysis import ErrorAnalysis
from scipy.stats import ttest_ind
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 2000)


def students_t_test(results1, results2):
    stat, p = ttest_ind(results1, results2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')


ea = ErrorAnalysis('base_best')

# SENTENCE LENGTH ANALYSIS

sentlen_dfs = [ea.compare_subsets(ea.w_preds, 'len', model) for model, _ in ea.models]
sentlen_df = ea.concat_comparisons(sentlen_dfs)
sentlen_df.index = ["0-90", "91-137", "138-192", "193-647", "All"]

print('Difference in performance depending on sentence length:')
print(sentlen_df.to_latex())

# IN QUOTE OR NOT + PRESENCE OF QUOTATION MARKS
quote_dfs = [ea.compare_subsets(ea.w_preds, 'quote', model, metrics=['rec', 'f1']) for model, _ in ea.models]
quote_df = ea.concat_comparisons(quote_dfs)

print('Difference in performance depending on whether in a quotes:')
print(quote_df.to_latex())

"""
df_for_quote_marks = ea.w_preds
for model, _ in ea.models:
    print()
    print(model)

    rate_of_quotes = sum(df_for_quote_marks.quote) / len(df_for_quote_marks)
    print(ea.N, rate_of_quotes)

    df_w_conf_mat = ea.conf_mat(df_for_quote_marks, model)

    for el in ['tp', 'fp', 'tn', 'fn']:
        subdf = df_for_quote_marks[df_for_quote_marks[el]]
        print(el)
        prop = sum(subdf.quote) / len(subdf)
        print(el, len(subdf), '\%' + str(round(prop*100,2)))
"""

# SOURCE AND STANCE
source_dfs = [ea.compare_subsets(ea.w_preds, 'source', model) for model, _ in ea.models]
source_df = ea.concat_comparisons(source_dfs)
print('Difference in performance depending on publisher:')
print(source_df.to_latex())

stance_dfs = [ea.compare_subsets(ea.w_preds, 'stance', model) for model, _ in ea.models]
stance_df = ea.concat_comparisons(stance_dfs)
print('Difference in performance depending on stance of article:')
print(stance_df.to_latex())

# LEXICAL BIAS

stance_dfs = [ea.compare_subsets(ea.w_preds, 'lex_bias', model) for model, _ in ea.models]
stance_df = ea.concat_comparisons(stance_dfs)
print('Difference in performance depending on presence of lexical bias:')
print(stance_df.to_latex())

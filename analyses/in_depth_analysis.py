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

sentlen_comparison = [ea.compare_subsets(ea.w_preds, 'len', model,  metrics=['f1']) for model, _ in ea.models]
sentlen_df = ea.concat_comparisons(sentlen_comparison)
sentlen_df.index = ["0-90", "91-137", "138-192", "193-647", "All"]

print('Difference in performance depending on sentence length:')
print(sentlen_df.to_latex())

# IN QUOTE OR NOT + PRESENCE OF QUOTATION MARKS
quote_comparison = [ea.compare_subsets(ea.w_preds, 'inf_quote', model, metrics=['red'], inf_bias_only=True) for model, _ in ea.models]
quote_df = ea.concat_comparisons(quote_comparison)

print('Difference in performance depending on annotated as in a quote:')
print(quote_df.to_latex())

quote_comparison = [ea.compare_subsets(ea.w_preds, 'auto_quote', model, metrics=['f1'], inf_bias_only=False) for model, _ in ea.models]
quote_df = ea.concat_comparisons(quote_comparison)

print('Difference in performance depending on whether contains quotation marks:')
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
source_comparison = [ea.compare_subsets(ea.w_preds, 'source', model, metrics=['f1']) for model, _ in ea.models]
source_df = ea.concat_comparisons(source_comparison)
source_df = source_df.loc[['fox', 'nyt', 'hpo', 'All']]
source_df.index = ['FOX', 'NYT', 'HPO', 'All']

print('Difference in performance depending on publisher:')
print(source_df.to_latex())

stance_comparison = [ea.compare_subsets(ea.w_preds, 'stance', model, metrics=['f1']) for model, _ in ea.models]
stance_df = ea.concat_comparisons(stance_comparison)
stance_df = stance_df.loc[['Right', 'Center', 'Left', 'All']]
print('Difference in performance depending on stance of article:')
print(stance_df.to_latex())

# LEXICAL BIAS

lex_comparison = [ea.compare_subsets(ea.w_preds, 'lex_bias', model, metrics=['prec', 'rec', 'f1']) for model, _ in ea.models]
lex_df = ea.concat_comparisons(lex_comparison)

print('Difference in performance depending on presence of lexical bias:')
print(lex_df.to_latex())

subj_comparison = [ea.compare_subsets(ea.w_preds, 'subj', model, metrics=['f1']) for model, _ in ea.models]
subj_df = ea.concat_comparisons(subj_comparison)
#subj_df.index = ["No", "Yes", "All"]
print(subj_df.to_latex())
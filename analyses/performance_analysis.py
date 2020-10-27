import pandas as pd
from lib.ErrorAnalysis import ErrorAnalysis
from scipy.stats import ttest_ind
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 2000)


ea = ErrorAnalysis('base_best')

# SENTENCE LENGTH ANALYSIS
sentlen_comparison = [ea.compare_subsets(ea.w_preds, 'len', model,  metrics=['f1']) for model, _ in ea.models]
sentlen_df = ea.concat_comparisons(sentlen_comparison)
#sentlen_df.index = ["0-90", "91-137", "138-192", "193-647", "All"]

print('Difference in performance depending on sentence length:')
print(sentlen_df.to_latex())

# IN QUOTE OR NOT
quote_comparison = [ea.compare_subsets(ea.w_preds, 'inf_quote', model, metrics=['rec'], inf_bias_only=True) for model, _ in ea.models]
quote_df = ea.concat_comparisons(quote_comparison)

print('Difference in performance depending on annotated as in a quote:')
print(quote_df.to_latex())

# SOURCE AND STANCE
source_comparison = [ea.compare_subsets(ea.w_preds, 'source', model, metrics=['f1']) for model, _ in ea.models]
source_df = ea.concat_comparisons(source_comparison)
source_df = source_df.loc[['fox', 'nyt', 'hpo', 'All']]
print('Difference in performance depending on publisher:')
print(source_df.to_latex())

print('Overlap between publisher and leaning:')
ct = pd.crosstab(ea.w_preds.source, ea.w_preds.stance, margins=True)
print(ct.loc[['fox', 'nyt', 'hpo'],['Right', 'Center', 'Left']])
publisher = ea.w_preds.source.replace({'fox': -1, 'nyt': 0, 'hpo': 1})
leaning = ea.w_preds.stance.replace({'Right': -1, 'Center': 0, 'Left': 1})
r = publisher.corr(leaning, method='spearman')
print('Correlation:', r)

stance_comparison = [ea.compare_subsets(ea.w_preds, 'stance', model, metrics=['f1']) for model, _ in ea.models]
stance_df = ea.concat_comparisons(stance_comparison)
stance_df = stance_df.loc[['Right', 'Center', 'Left', 'All']]
print('Difference in performance depending on leaning of article:')
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
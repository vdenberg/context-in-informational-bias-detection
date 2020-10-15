import pandas as pd
from lib.ErrorAnalysis import ErrorAnalysis
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 2000)

ea = ErrorAnalysis('base_best')

sentlen_dfs = [ea.compare_subsets(ea.w_preds, 'len', model, context) for model, context in ea.models]
sentlen_df = ea.concat_comparisons(sentlen_dfs)
sentlen_df.index = ["0-90", "91-137", "138-192", "193-647", "All"]
print(sentlen_df.to_latex())

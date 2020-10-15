import pandas as pd
from lib.handle_data.ErrorAnalysis import ErrorAnalysis
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 200)

ea = ErrorAnalysis(models='base_best')

source_dfs = [ea.compare_subsets(ea.w_preds, 'source', model, context) for model, context in ea.models]
source_df = ea.concat_comparisons(source_dfs)
print(source_df.to_latex())

stance_dfs = [ea.compare_subsets(ea.w_preds, 'stance', model, context) for model, context in ea.models]
stance_df = ea.concat_comparisons(stance_dfs)
print(stance_df.to_latex())




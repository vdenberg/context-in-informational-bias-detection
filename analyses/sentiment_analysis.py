import pandas as pd
from lib.ErrorAnalysis import ErrorAnalysis
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 200)

ea = ErrorAnalysis(models='base_best')

sentlen_dfs = [ea.compare_subsets(ea.w_preds, 'subj', model, context) for model, context in ea.models]
sentlen_df = ea.concat_comparisons(sentlen_dfs)
# sentlen_df.index = ["0", "1-5.26", "5.37-8.57", "8.58-13.51", "13.52-66.67", "All"]
#sentlen_df.index = ["0", "0-3.7", "3.8-9.52", "9.53-66-67", "All"]
#sentlen_df.index = ["None", "Weak", "Normal", "Strong", "All"]
sentlen_df.index = ["No", "Yes", "All"]
print(sentlen_df)
sentlen_df = sentlen_df[['N', '%Bias', 'F1_rob_22', 'F1_cim_coverage']]
print(sentlen_df.to_latex())

pol_df = ea.clean_for_pol_analysis()
pol_dfs = [ea.compare_subsets(pol_df, 'inf_pol', model, context) for model, context in ea.models]
pol_df = ea.concat_comparisons(pol_dfs, only_rec=True)

print(pol_df.to_latex())

dir_df = ea.clean_for_dir_analysis()
dir_dfs = [ea.compare_subsets(dir_df, 'inf_dir', model, context) for model, context in ea.models]
dir_df = ea.concat_comparisons(dir_dfs, only_rec=True)
print(dir_df.to_latex())

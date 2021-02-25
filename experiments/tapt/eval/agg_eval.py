import argparse, os, json
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-minval', '--minval', type=float, default=0.75, help='minimum validation value for jackknifing')
    parser.add_argument('-dirs', '--results_dirs', nargs="+", type=str, default=['../../hp_reproduction/re_roberta_hp_515_ft_results'], help='results file name')
    args = parser.parse_args()

    VAL_CUTOFF = args.minval

    locations = args.results_dirs
    for loc in locations:
        result_dirn = os.path.basename(loc.strip('/'))
        info = result_dirn[:-len('_ft_results')]
        info = info.split('_')
        model, split_type = '_'.join(info[:-2]), '_'.join(info[-2:])

        agg = []
        for fn in os.listdir(loc): #re_roberta_hp_515_ft_results
            seed = fn.split('_')[-1]
            fold = fn.split('_')[-2] #'_'.join(fn.split('_')[1:-1])
            met_fn = os.path.join(fn, 'metrics.json')
            met_fp = os.path.join(loc, met_fn)
            if os.path.exists(met_fp):
                with open(met_fp, 'r') as f:
                    mets = json.load(f)
                    mets.update({'model': model, 'seed': seed, 'fold': fold})
                    agg.append(mets)

        # split df
        all_df = pd.DataFrame(agg)
        best_val_df = all_df[all_df.best_validation_f1 > VAL_CUTOFF]

        pd.set_option('display.max_rows', 500)
        #print(all_df.groupby(['seed', 'fold'])['best_validation_f1'].mean())


        # format interesting col
        test_col = [i for i in all_df.columns if 'test' in i]
        int_col = ['seed', 'test_f1']
        all_df[test_col] = all_df[test_col].round(4) * 100
        best_val_df[test_col] = best_val_df[test_col].round(4) * 100

        # m and std
        all_descr = all_df[int_col].groupby('seed').mean().describe()
        test_m = all_descr.loc['mean'].round(2).astype(str)
        test_std = all_descr.loc['std'].round(2).astype(str)
        all_result = test_m + ' +- ' + test_std + f' ({len(all_df.seed.unique())} seeds) ({len(all_df.fold.unique())} folds)'

        print(f"--- \n{model} {split_type} results ---")
        print("\nAll results:")
        print(all_result)

        best_val_descr = best_val_df[int_col].groupby('seed').mean().describe()
        test_m = best_val_descr.loc['mean'].round(2).astype(str)
        test_std = best_val_descr.loc['std'].round(2).astype(str)
        best_val_result = test_m + ' +- ' + test_std + f' ({len(best_val_df.seed.unique())} seeds) ({len(all_df.fold.unique())} folds)'

        print(f"\nResults if best_val > {VAL_CUTOFF}:")
        print(best_val_result)



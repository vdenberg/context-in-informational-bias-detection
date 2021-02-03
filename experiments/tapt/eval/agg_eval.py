import argparse, os, json
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '--results_dir', type=str, default='../../hp_reproduction/re_roberta_hp_515_ft_results', help='results file name')
    args = parser.parse_args()

    results_dir = args.results_dir
    result_dirn = os.path.basename(results_dir)
    info = result_dirn[:-len('_ft_results')]
    info = info.split('_')
    model, split_type = '_'.join(info[:-2]), '_'.join(info[-2:])

    agg = []
    for fn in os.listdir(results_dir): #re_roberta_hp_515_ft_results
        seed = fn.split('_')[-1]
        fold = '_'.join(fn.split('_')[1:-1])
        met_fp = os.path.join(fn, 'metrics.json')
        with open(os.path.join(results_dir, met_fp), 'r') as f:
            mets = json.load(f)
            mets.update({'model': model, 'seed': seed, 'fold': fold})
            agg.append(mets)

    # split df
    all_df = pd.DataFrame(agg)
    best_val_df = all_df[all_df.best_validation_f1 > 0.75]

    # format interesting col
    test_col = [i for i in all_df.columns if 'test' in i]
    int_col = ['seed', 'test_f1']
    all_df[test_col] = all_df[test_col].round(4) * 100
    best_val_df[test_col] = best_val_df[test_col].round(4) * 100

    # m and std
    all_descr = all_df[int_col].groupby('seed').mean().describe()
    test_m = all_descr.loc['mean'].round(2).astype(str)
    test_std = all_descr.loc['std'].round(2).astype(str)
    all_result = test_m + ' +- ' + test_std + f' ({len(all_df.seed.unique())} seeds)'

    print(f"\n{model} {split_type} results:")
    print(all_result)

    #best_val_descr = best_val_df[int_col].groupby('seed').mean().describe()
    #test_m = best_val_descr.loc['mean'].round(2).astype(str)
    #test_std = best_val_descr.loc['std'].round(2).astype(str)
    #best_val_result = test_m + ' +- ' + test_std + f' ({len(best_val_df.seed.unique())} seeds)'

    #print(f"\n{model} {split_type} results if best_val > .75:")
    #print(best_val_result)



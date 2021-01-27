import argparse, os, json
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '--results_dir', type=str, default='hp_reproduction/re_roberta_hp_515_ft_results', help='results file name')
    args = parser.parse_args()

    results_dir = args.results_dir
    model = '_'.join(results_dir.split('_')[:-2])

    agg = []
    for fn in os.listdir(results_dir): #re_roberta_hp_515_ft_results
        seed = fn.split('_')[-1]
        met_fp = os.path.join(fn, 'metrics.json')
        with open(os.path.join(results_dir, met_fp), 'r') as f:
            mets = json.load(f)
            mets.update({'model': model, 'seed': seed})
            agg.append(mets)

    # split df
    all_df = pd.DataFrame(agg)
    best_val_df = all_df[all_df.best_validation_f1 > 0.75]

    # interesting col
    test_col = [i for i in all_df.columns if 'test' in i]
    # test_f1_col = [i for i in test_col if 'f1' in i]

    # m and std
    all_descr = all_df[test_col].mean().describe()
    test_m = all_descr.loc['mean'].round(2).astype(str)
    test_std = all_descr.loc['std'].round(2).astype(str)
    all_result = test_m + ' +- ' + test_std

    best_val_descr = best_val_df[test_col].mean().describe()
    print(best_val_df[test_col])
    print(best_val_descr)
    test_m = best_val_descr.loc['mean'].round(2).astype(str)
    test_std = best_val_descr.loc['std'].round(2).astype(str)
    best_val_result = test_m + ' +- ' + test_std

    print(f"\n{model} results:")
    print(all_result)
    print(f"\n{model} results if best_val > .75:")
    print(best_val_result)



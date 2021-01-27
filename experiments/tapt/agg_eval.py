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

    agg_df = pd.DataFrame(agg)
    f1_col = [i for i in agg_df.columns if 'f1' in i]
    #agg_df = agg_df[agg_df.best_validation_f1 > 0.80]

    print(agg_df.mean())



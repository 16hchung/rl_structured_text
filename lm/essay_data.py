import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

# import all data
def get_all_data(fname='lm/training_set_rel3.tsv'):
    df = pd.read_csv(fname, sep='\t', header=0, encoding='latin1')
    # only want essayset = 2
    df = df[df['essay_set'] == 2]
    return df

def scores_from_df(df):
    domain_cols = ['domain1_score', 'domain2_score']
    scores = df[domain_cols].to_numpy()
    scores = np.average(scores, axis=1)
    return scores

# histogram scores
def gen_histogram(scores, fig_save='figures/scores_hist.png'):
    plt.hist(scores, bins=10)
    plt.savefig(fig_save)

# quality threshold
def thresh_quality(df, scores, df_save = 'filtered_df.tsv', txt_save='filtered_train.txt'):
    par_dir = 'lm/data/'
    txt_save = par_dir + txt_save
    df_save = par_dir + df_save
    filtered = df[scores >= 3]
    # save text only
    text = list(filtered['essay'])
    with open(txt_save, 'w') as f:
        for t in text:
            f.write(t)
            f.write('\n\n')
    #filtered['essay'].to_csv(txt_save, escapechar='\\', quotechar='', index=False, header=False, quoting=csv.QUOTE_NONE) 
    # save dataframe with more info
    filtered.to_csv(df_save, sep='\t')
    return filtered

# save figures + filtered data file


if __name__=='__main__':
    df = get_all_data()
    scores = scores_from_df(df)
    #gen_histogram(df)
    thresh_quality(df, scores)

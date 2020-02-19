import pandas as pd
import numpy as np
import csv
import math
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
def thresh_quality(df, scores, df_save = 'filtered_df.tsv', txt_save='filtered/'):
    par_dir = 'lm/data/'
    txt_save = par_dir + txt_save
    df_save = par_dir + df_save
    filtered = df[scores >= 3]
    # save train, test, dev text only
    text = list(filtered['essay'])
    train, test, dev = split_data(text)
    def save_txt(data_type, dataset):
        with open(txt_save + data_type, 'w') as f:
            for t in dataset:
                f.write(t)
                f.write('\n\n')
    #filtered['essay'].to_csv(txt_save, escapechar='\\', quotechar='', index=False, header=False, quoting=csv.QUOTE_NONE) 
    save_txt('train.txt', train)
    save_txt('test.txt', test)
    save_txt('dev.txt', dev)

    # save dataframe with more info
    filtered.to_csv(df_save, sep='\t')
    return filtered

# save figures + filtered data file

# randomly split data into train, test, dev sets
def split_data(text):
    import pdb;pdb.set_trace()
    length = len(text)
    indices = np.linspace(0, length - 1, num = length - 1, dtype=int)
    np.random.shuffle(indices)
    train_split = math.ceil(length * .9)
    test_split = math.ceil(length * .95)
    train_indices = indices[:train_split]
    test_indices = indices[train_split:test_split]
    dev_indices = indices[test_split:]
    train = [text[i] for i in train_indices]
    test = [text[i] for i in test_indices]
    dev = [text[i] for i in dev_indices]
    return train, test, dev


if __name__=='__main__':
    df = get_all_data()
    scores = scores_from_df(df)
    #gen_histogram(df)
    thresh_quality(df, scores)

'''
Files that we want to generate:
- df with sentence info: sentence id, sentence text, essay id, sentence idx in essay, sentence embedding
- df with random sentence pair idxs: sentence id 0, sentence id 1, sequential?
- df with features: sentence id 0, sentece id 1, embedding 0 concat embedding 1, sequential ?
'''
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from random import randint

from util import paths as p

# TODO: maybe move into some constants file
essay_id_key = 'essay_id'
sentence_txt_key = 'sentence_txt'
idx_in_essay_key = 'idx_in_essay'
sentence_id_key = 'sentence_id'
sentence_id_key0 = sentence_id_key+'0'
sentence_id_key1 = sentence_id_key+'1'
sequential_key = 'sequential'
pairs_id_key = 'pairs_id'

def extract_sentences(essay_df):
    # constants
    cols = [essay_id_key, sentence_txt_key, idx_in_essay_key] # once done: , 'embedding']
    
    sent_df = pd.DataFrame()
    for idx, essay_row in essay_df.iterrows():
        # split essay into sentences
        sentences = re.split('[;\.\?!] +', essay_row['essay'])
        sentences = [s for s in sentences if len(s) > 0]
        idxs = list(range(len(sentences)))
        # set essay id, sentence idx
        data = {
            sentence_txt_key: sentences, 
            idx_in_essay_key: idxs,
            essay_id_key: essay_row[essay_id_key]
        }
        this_essay_df = pd.DataFrame(data, columns=cols)
        # TODO add embedding field
        sent_df = sent_df.append(this_essay_df, ignore_index=True)
        
    sent_df.to_csv(p.LMData.sentences_df, index_label=sentence_id_key, sep='\t')
    return sent_df

def gen_sentence_pairs(sent_df):
    cols = [sentence_id_key0, sentence_id_key1, sequential_key]
    n_sentences = sent_df.shape[0]
    n_pairs = n_sentences - 1

    def data_for_idcs(idx0, idx1, seq):
        data = {
            sentence_id_key0: sent_df[sentence_id_key][idx0],
            sentence_id_key0: sent_df[sentence_id_key][idx1],
            sequential_key: seq
        }
        return data

    # generate positive labels
    pairs_df = pd.DataFrame()
    curr_essay_id = -1
    for idx, sent_row in tqdm(sent_df.iterrows()):
        if curr_essay_id != sent_row[essay_id_key]:
            continue
        data = data_for_idcs(idx-1, idx, True)
        this_pair_df = pd.DataFrame(data, columns=cols)
        pairs_df = pairs_df.append(this_pair_df, ignore_index=True)

    # generate negative labels
    n_false_generated = 0
    while n_false_generated < n_pairs:
        zeroth = randint(0, n_pairs-1)
        first  = randint(0, n_pairs-1)
        zero_essay_id  = sent_df[essay_id_key][zeroth]
        first_essay_id = sent_df[essay_id_key][first]
        if first == zeroth + 1 and zero_essay_id == first_essay_id: # sequential
            continue
        data = data_for_idcs(zeroth, first, False)
        this_pair_df = pd.DataFrame(data, columns=cols)
        pairs_df = pairs_df.append(this_pair_df, ignore_index=True)
        n_false_generated += 1
    
    pairs_df.to_csv(p.LMData.pairs_df, index_label=pairs_id_key, sep='\t')
    return pairs_df

if __name__=='__main__':
    df = pd.read_csv(p.LMData.thresh_df, sep='\t')
    sent_df = extract_sentences(df)
    pairs_df = gen_sentence_pairs(sent_df)






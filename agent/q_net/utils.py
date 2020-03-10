from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import csv
import pandas as pd

# GLOBAL
prompt = 'Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive?'

def scores_from_df(df):
    domain_cols = ['domain1_score', 'domain2_score']
    scores = df[domain_cols].to_numpy()
    scores = np.ceil(np.average(scores, axis=1))
    return scores

TOT = 5
def sample_per_score(df, essays):
    for s in range(2):
        s1 = np.array((essays[s][0])['essay_id'].values)
        s2 = np.array((essays[s][1])['essay_id'].values)
        s3 = np.array((essays[s][2])['essay_id'].values)
        choices = np.concatenate((s1,s2,s3), -1)
        #idx = np.random.choice(data['essay_id'], size=TOT,replace=True)
        idx = np.random.choice(choices, size=TOT, replace=True)
        batch = []
        for i in idx:
            batch.append(df[df['essay_id'] == i].get(['essay'])) #.values[0,0])
            #batch.append(data[data['essay_id'] == i].get(['essay']))
        yield batch

EXPATH = '/home/nlp/rl_structured_text/lm/training_set_rel3.tsv'
def evaluator(df, essays, model, tokenizer, sample, device):
    tokenized = tokenizer.encode(sample)
    output = torch.tensor([tokenized])
    model.eval()
    model.to(device)
    model.cuda()
    _, _, hiddens = model(output.to(device), past=None)
    our_emb = hiddens[-1][:,-1,:]
    scores_list = []
    batches = []
    for batch in sample_per_score(df, essays):
        scores = []
        #scores_s1 = []
        #scores_s2 = []
        #s1_tokenized = tokenizer.encode(sample1.values[0][0][:1024])
        #s2_tokenized = tokenizer.encode(sample2.values[0][0][:1024])
        #s1 = torch.tensor([s1_tokenized])
        #s2 = torch.tensor([s2_tokenized])
        #_, _, sample_h1 = model(s1.to(device), past=None)
        #_, _, sample_h2 = model(s2.to(device), past=None)
        #sample_h1_emb = sample_h1[-1][:,-1,:]
        #sample_h2_emb = sample_h2[-1][:,-1,:]

        for essay in batch:
            #essay_tokenized = tokenizer.encode(essay[:1024])
            essay_tokenized = tokenizer.encode(essay.values[0][0][:1024])
            essay_output = torch.tensor([essay_tokenized])
            _, _, sample_hiddens = model(essay_output.to(device), past=None)
            sample_emb = sample_hiddens[-1][:,-1,:]
            sim = our_emb.squeeze(0) @ sample_emb.squeeze(0)
            #scores_s1.append((sample_h1_emb.squeeze(0) @ sample_emb.squeeze(0)).item())
            #scores_s2.append((sample_h2_emb.squeeze(0) @ sample_emb.squeeze(0)).item())
            scores.append(sim.item())
        scores = np.array(scores)
        score = np.mean(scores)
        scores_list.append(score)
    nearest = np.argmax(scores_list)
    return nearest + 1

def get_essay_samples():
    df = pd.read_csv(EXPATH, sep='\t', header=0, encoding='latin1')
    df = df[df['essay_set'] == 2]
    scores = scores_from_df(df)
    essays = []
    score_range = np.linspace(1.0, 6.0, num=6)
    for score in score_range:
        essays.append(df[scores==score])
    return essays

def get_essay_samples1():
    df = pd.read_csv(EXPATH, sep='\t', header=0, encoding='latin1')
    df = df[df['essay_set'] == 2]
    scores = scores_from_df(df)
    essays = []
    sampled_essays = []
    #score_range = np.linspace(1.0, 6.0, num=6)
    score_range = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    for score in score_range:
        essays.append([df[scores==s][:-2] for s in score])
        print('====:)====')
        essay1 = df[scores==1.0][-2:-1]['essay'].values[0] 
        essay2 = df[scores==5.0][-1:]['essay'].values[0]
        sampled_essays.append((essay1, essay2))
    return df, essays, sampled_essays

device = torch.device('cuda')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config()
config.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
model.eval()
model.to(device)
model.cuda()

df, SAMPLES, SEQUENCES = get_essay_samples1()
true_label = 1
for (seq1, seq2) in SEQUENCES:
    print('true', true_label)
    r_score1 = evaluator(df, SAMPLES, model, tokenizer, seq1, device)
    print(r_score1)
    r_score2 = evaluator(df, SAMPLES, model, tokenizer, seq2, device)
    print(r_score2)
    print('====')
    true_label += 1

#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#config = GPT2Config()
#config.output_hidden_states = True

def rand_gen_first_token(model, tokenizer, device):
    #prompt = 'Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive?'
    okay_tokens = [1002 , 314  , 1471 , 2141 , 383  , 843  , 1867 , 632  , 921  , 1148 , 4231 , 4362 , 770  , 554  , 
                   1320 , 1374 , 775  , 1400 , 4162 , 4222 , 366  , 10928, 317  , 10358, 3363 , 8192 , 1649 , 1680 , 
                   1114 , 1081 , 8314 , 2312 , 1318 , 1675 , 23998, 1119 , 3406 , 3226 , 4418 , 3914 , 7731 , 1892 , 
                   3894 , 3244 , 4377 , 2011 , 6350 , 8673 , 4619 , 1406 , 3412 , 10347, 2773 , 2293 , 887  , 1881 , 
                   28416, 352  , 2750 , 6952 , 1629 , 8013 , 1550 , 2329 , 1703 , 5845 , 4784 , 4650 , 6674 , 4042 , 
                   1052 , 3274 , 1439 , 2561 , 362  , 4900 , 16263, 5338 , 15176, 25110, 7875 , 2893 , 9022 , 15323, 
                   3423 , 3954 , 2080 , 18948, 4149 , 2094 , 4091 , 4525 , 4380 , 513  , 14026, 6521 , 28417, 43048, 
                   3574 , 6930 , 23631, 1355 , 18578, 9170 , 11399, 2102 , 1770 , 8447 , 2735 , 5514 , 367  , 838  , 
                   767  , 16981, 3582 , 642  , 10127, 6914 , 9506 , 7214 , 604  , 8989 , 8920 , 12642]
    # TODO do we wanna include this as a state/action pair? or just as random init?
    #      ie word we generate here is s_0
    # start state is beginning of seq token (actually same as end of seq token)
    generated = tokenizer.encode(prompt)
    context = torch.tensor([generated]) #, device=torch.device('cpu'))
    # do one iteration but not greedy to generate random token
    model.to(device)
    model.cuda()
    logits, past, hiddens= model(context.to(device), past=None)
    token = None
    if len(hiddens[-1].shape) > 2:
        state = hiddens[-1][:,-1,:]
    else:
        state = hiddens[-1]
    while token not in okay_tokens:
        topk = torch.topk(F.softmax(logits[...,-1,:], 1),150)
        token_idx = torch.multinomial(topk.values, 1).item()
        token = topk.indices[0,token_idx].item()
    return [token], past, state

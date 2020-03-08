from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from utils import rand_gen_first_token
import torch
import torch.nn
import torch.distributions
import random
import numpy as np
from policy import Policy
import pickle
import math
import h5py
import argparse

# CONSTANTS
N_EPOCH = int(1e3)
LOG_FREQ = 2
EPIS_PER_EPOCH = 5
MAX_BUF = 1e4
MIN_LENGTH = 100
END_TOKEN = '<|endoftext|>'
EPS_DECAY = .9
N_ACTIONS = 5
STATE_SZ = 768
BATCH_SZ = 100
N_BATCHES = 5 
MAX_PATH = 1e4
SAVEPATH = 'policy_network.bin'
gamma = .99
learning_rate = .01
fname = 'PG_generated.txt'
MAX_LENGTH = 994 #1024
FILEPATH = 'PG'

paths = []

def init_models():
    import pdb; pdb.set_trace()
    
    #device = torch.device('cuda')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = GPT2Config()
    config.output_hidden_states = True
    
    model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
    model.to(device)
    model.cuda()

    policy = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
    policy.to(device)
    model.cuda()

    # finetune model as policy net
    optimizer = torch.optim.RMSprop(model.lm_head.parameters(), lr = learning_rate)
    
    return device, tokenizer, model

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--r_prob_scaler', default=1.)
    parser.add_argument('--r_tgt_word_scaler', default=1.)
    parser.add_argument('--r_simscore_scaler', default=0.5)
    cmd_args = parser.parse_args()

    target_words = torch.load('target_words.pt').tolist()
    device, tokenizer, model = init_models()

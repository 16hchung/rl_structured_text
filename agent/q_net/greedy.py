from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import torch.nn
import torch.nn.functional as F
import torch.distributions
import random
import numpy as np
import pickle
from utils import rand_gen_first_token
import h5py

END_TOKEN = '<|endoftext|>'
fname = 'greedy.txt'
MAX_LENGTH = 994
N_EPS = 100
FILEPATH = 'greedy'

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config()
config.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
#device = torch.device("cuda" if args.cuda else "cpu")

def gen_episode(outf, f, eps):
    model.eval()
    reward_sum = 0.0
    data = f.create_group('eps' + str(eps))
    dt = h5py.string_dtype(encoding='ascii')
    data.create_dataset('final_reward', (1,), dtype='f')
    data.create_dataset('final_length', (1,), dtype='i')
    data.create_dataset('state', (MAX_LENGTH, ), dtype=dt) #, dtype='S')
    #data.create_dataset('emb_state', (MAX_LENGTH, 768), dtype='f')
    data.create_dataset('action', (MAX_LENGTH, ), dtype='i')
    data.create_dataset('prob', (MAX_LENGTH, ), dtype='f')
    data.create_dataset('reward', (MAX_LENGTH, ), dtype='f')
    generated, past = rand_gen_first_token(model, tokenizer, device=None)
    f['eps' + str(eps)]['state'][0] = tokenizer.decode(generated[-1])
    #f['eps' + str(eps)]['emb_state'][0] = past
    length = 0
    context = torch.tensor([generated])
    while generated[-1] != tokenizer.encode([END_TOKEN])[0] and length < MAX_LENGTH-1:
        logits, past, _ = model(context, past=past)
        token = torch.argmax(logits[..., -1, :])
        generated += [token.tolist()]
        context = token.unsqueeze(0)
        sequence = tokenizer.decode(generated)
        f['eps' + str(eps)]['state'][length+1] = tokenizer.decode(generated[-1])
        #f['eps' + str(eps)]['emb_state'][length+1] = past
        f['eps' + str(eps)]['action'][length] = 0 # always make the greedy choice
        if len(F.softmax(logits[...,-1,:]).shape) == 2:
            f['eps' + str(eps)]['prob'][length] = F.softmax(logits[...,-1,:])[:,token].item()
            f['eps' + str(eps)]['reward'][length] = F.softmax(logits[...,-1,:])[:,token].item()
            reward_sum += F.softmax(logits[...,-1,:])[:,token].item()
        else:
            f['eps' + str(eps)]['prob'][length] = F.softmax(logits[...,-1,:])[token].item()
            f['eps' + str(eps)]['reward'][length] = F.softmax(logits[...,-1,:])[token].item()
            reward_sum += F.softmax(logits[...,-1,:])[token].item()
        length += 1
        #print(sequence)
        #print('\n')
    outf.write(sequence)
    f['eps'+str(eps)]['final_reward'][0] = reward_sum
    f['eps'+str(eps)]['final_length'][0] = length
    if length >= MAX_LENGTH:
        outf.write(END_TOKEN)

with open(fname, 'w') as outf:
    f = h5py.File(FILEPATH + '.hdf5', 'w')
    for i in range(N_EPS):
        gen_episode(outf, f, i)
    f.close()


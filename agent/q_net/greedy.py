from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import torch.nn
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
    import pdb;pdb.set_trace()
    data = f.create_group('eps' + str(eps))
    data.create_data('state', (MAX_LENGTH, ), dtype='S')
    data.create_data('emb_state', (MAX_LENGTH, 768), dtype='f')
    data.create_data('action', (MAX_LENGTH, ), dtype='i')
    data.create_data('prob', (MAX_LENGTH, ), dtype='f')
    data.create_data('reward', (MAX_LENGTH, ), dtype='f')
    generated, past = rand_gen_first_token(model, tokenizer, device=None)
    f['eps' + str(eps)]['state'][0] = generated[-1]
    f['eps' + str(eps)]['emb_state'][0] = past
    length = 0
    context = torch.tensor([generated])
    while generated[-1] != tokenizer.encode([END_TOKEN])[0] and length < MAX_LENGTH-1:
        logits, past, _ = model(context, past=past)
        token = torch.argmax(logits[..., -1, :])
        generated += [token.tolist()]
        context = token.unsqueeze(0)
        sequence = tokenizer.decode(generated)
        f['eps' + str(eps)]['state'][length+1] = generated[-1]
        f['eps' + str(eps)]['emb_state'][length+1] = past
        f['eps' + str(eps)]['action'][length] = 0 # always make the greedy choice
        f['eps' + str(eps)]['prob'][length] = logits[...,-1,token]
        f['eps' + str(eps)]['reward'][length] = F.softmax(logits[...,-1,token])
        length += 1
        #print(sequence)
        #print('\n')
    outf.write(sequence)
    if length == MAX_LENGTH:
        outf.write(END_TOKEN)

with open(fname, 'w') as outf:
    f = h5py.File(FILEPATH + '.hdf5', 'w')
    for i in range(N_EPS):
        gen_episode(outf, f, i)
    f.close()


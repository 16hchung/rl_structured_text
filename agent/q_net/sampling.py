from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import torch.nn
import torch.distributions
import torch.nn.functional as F
import random
import numpy as np
import pickle
from utils import rand_gen_first_token
import h5py

END_TOKEN = '<|endoftext|>'
fname = 'sampling.txt'
MAX_LENGTH = 994
N_EPS = 100
FILEPATH = 'sampling'

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config()
config.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
#device = torch.device("cuda" if args.cuda else "cpu")

def gen_episode(outf, f, eps):
    model.eval()
    data = f.create_group('eps' + str(eps))
    data.create_data('state', (MAX_LENGTH,), dypte='S')
    data.create_data('emb_state', (MAX_LENGTH, 768), dypte='f')
    data.create_data('action', (MAX_LENGTH,), dtype='i')
    data.create_data('prob', (MAX_LENGTH,), dtype='f')
    data.create_data('reward', (MAX_LENGTH,), dypte='f')
    generated, past = rand_gen_first_token(model, tokenizer, device=None)
    f['eps' + str(eps)]['state'][0] = generated[-1]
    f['eps' + str(eps)]['emb_state'][0] = past
    length = 0
    context = torch.tensor([generated])
    while generated[-1] != tokenizer.encode([END_TOKEN])[0] and length < MAX_LENGTH-1:
        logits, past, _ = model(context, past=past)
        if len(logits.shape) > 2:
            topk = torch.topk(F.softmax(logits[...,-1,:], 1),100)
        else:
            topk = torch.topk(F.softmax(logits, 1), 100)
        token_idx = torch.multinomial(topk.values, 1).item()
        token = topk.indices[0,token_idx].item()
        generated += [token]
        context = torch.tensor([token])
        sequence = tokenizer.decode(generated)
        f['eps' +str(eps)]['state'][length+1] = generated[-1]
        f['eps' + str(eps)]['emb_state'][length+1] = past
        f['eps' + str(eps)]['action'][length] = token_idx
        f['eps' + str(eps)]['prob'][length] = F.softmax(logits[...,-1,token])
        length += 1
    outf.write(sequence)
    if length == MAX_LENGTH:
        outf.write(END_TOKEN)

with open(fname, 'w') as outf:
    f = h5py.File(FILEPATH + '.hdf5', 'w')
    for i in range(N_EPS):
        gen_episode(outf, f, i)
    f.close()


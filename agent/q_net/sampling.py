from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import torch.nn
import torch.distributions
import torch.nn.functional as F
import random
import numpy as np
import pickle
from utils import rand_gen_first_token


END_TOKEN = '<|endoftext|>'
fname = 't_sampling.txt'
MAX_LENGTH = 994
N_EPS = 100

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config()
config.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
#device = torch.device("cuda" if args.cuda else "cpu")

def gen_episode(outf):
    model.eval()
    generated, past = rand_gen_first_token(model, tokenizer, device=None)
    length = 1
    context = torch.tensor([generated])
    while generated[-1] != tokenizer.encode([END_TOKEN])[0] and length < MAX_LENGTH:
        logits, past, _ = model(context, past=past)
        if len(logits.shape) > 2:
            topk = torch.topk(F.softmax(logits[...,-1,:], 1),100)
        else:
            topk = torch.topk(F.softmax(logits, 1), 100)
        token_idx = torch.multinomial(topk.values, 1).item()
        token = topk.indices[0,token_idx].item()
        length += 1
        generated += [token]
        context = torch.tensor([token])
        sequence = tokenizer.decode(generated)
    outf.write(sequence)
    if length == MAX_LENGTH:
        outf.write(END_TOKEN)

with open(fname, 'w') as outf:
    for i in range(N_EPS):
        gen_episode(outf)

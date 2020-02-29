from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import torch.nn
import torch.distributions
import random
import numpy as np
import pickle
from utils import rand_gen_first_token


END_TOKEN = '<|endoftext|>'
fname = 'greedy.txt'
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
        token = torch.argmax(logits[..., -1, :])
        length += 1
        generated += [token.tolist()]
        context = token.unsqueeze(0)
        sequence = tokenizer.decode(generated)
        #print(sequence)
        #print('\n')
    outf.write(sequence)
    if length == MAX_LENGTH:
        outf.write(END_TOKEN)

with open(fname, 'w') as outf:
    for i in range(N_EPS):
        gen_episode(outf)

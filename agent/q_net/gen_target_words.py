from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

prompt = 'Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive?'

device = torch.device("cuda")# if args.cuda else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config()
config.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)

model.to(device)
model.cuda()

import pdb;pdb.set_trace()
text_idx = tokenizer.encode(prompt)
word_vectors = model.transformer.wte.weight[text_idx,:]
avg_vec = word_vectors.mean(dim=0, keepdims=True)
distances = model.transformers.wte.weight @ avg_vec.T

topk = torch.topk(distances, 100)
torch.save(topk.indices, 'target_words. pt')

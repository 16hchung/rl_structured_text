from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import random

#from .model import QNet

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config()
config.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)

generated = tokenizer.encode("The Manhattan Bridge")
context = torch.tensor([generated])
past = None

for i in range(100):
    print(i)
    output = model(context, past=past)
    #output, past = model(context, past=past)
    logits = output[0]
    hiddens = output[2]
    q_state = hiddens[-1][:,-1,:]
    #tokens = torch.argmax(logits[..., -1, :])
    idx = torch.topk(logits[:,-1,:], k=5,dim=-1)[1]
    token = idx[0]
    generated += [token.tolist()]
    context = token.unsqueeze(0)

sequence = tokenizer.decode(generated)

print(sequence)


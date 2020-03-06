from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from utils import rand_gen_first_token
import torch
import torch.nn
import torch.distributions
import random
import numpy as np
from policy import Policy
import pickle

# CONSTANTS
N_EPOCH = int(1e3)
LOG_FREQ = 2
EPIS_PER_EPOCH = 5
MAX_BUF = 1e4
END_TOKEN = '<|endoftext|>'
EPS_DECAY = .9
N_ACTIONS = 5
STATE_SZ = 768
BATCH_SZ = 100
N_BATCHES = 5 
MAX_PATH = 1e4
SAVEPATH = 'policy_debug.bin'
gamma = .9
learning_rate = .01
fname = 'PG_debug.txt'
MAX_LENGTH = 994 #1024

paths = []

#device = torch.device("cuda" if args.cuda else "cpu")


''' GPT 2 GENERATING CODE '''
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config()
config.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)


''' POLICY TRAINING CODE '''
policy = Policy()

#policy.model.load_state_dict(torch.load(SAVEPATH))

optimizer = torch.optim.RMSprop(policy.parameters(), lr=learning_rate)

eps = .9

def get_returns(rewards):
    returns = np.zeros(len(rewards))
    for i in range(len(rewards)):
        r = rewards[i:]
        discount = [gamma ** t for t in range(len(r))]
        returns[i] = np.array(r) @ np.array(discount)
    return returns

def gen_episode(outf, epi):
    with torch.no_grad():
        model.eval()
        path = {}
        path['state'] = []
        path['action'] = []
        path['reward'] = []
        policy.eval() # freeze
        generated, past = rand_gen_first_token(model, tokenizer, device=None)
        context = torch.tensor([generated])
        sequence = tokenizer.decode(generated)
        length = 1
        # generate trajectory for episode
        while generated[-1] != tokenizer.encode([END_TOKEN])[0] and length < MAX_LENGTH:
            logits, past, hiddens = model(context, past=past)
            #logits = output[0]
            #past = output[1]
            #hiddens = output[2]
            if len(hiddens[-1].shape) > 2:
                state = hiddens[-1][:,-1,:]
            else:
                state = hiddens[-1]
            action_logits = policy(state)
            import pdb;pdb.set_trace()
            rewards, idx = torch.topk(logits[...,-1,:], k=5, dim=-1)
            m = torch.distributions.Categorical(action_logits)
            action = m.sample().item()
            rewards = rewards.squeeze(0)
            reward = torch.softmax(rewards, -1)[action]
            idx = idx.squeeze(0)
            token = idx[action]
            generated += [token.tolist()]
            length += 1
            context = token.unsqueeze(0)
            sequence = tokenizer.decode(generated)
            path['state'].append(state)
            path['action'].append(action)
            path['reward'].append(reward.data.numpy())
        outf.write(sequence)
        if length == MAX_LENGTH:
            path['reward'][-1] -= 100
        paths.append(path)


losses = []
with open(fname, 'w') as outf:
    for epoch in range(N_EPOCH):
        for epi in range(EPIS_PER_EPOCH):
            gen_episode(outf, epi)
            outf.write('\n')
            outf.write('\n')
        policy.train()
        cum_loss = 0
        for path in paths:
            optimizer.zero_grad()
            states = path['state']
            actions = path['action']
            rewards = path['reward']
            states = torch.cat(states)
            rewards = np.array(rewards)
            action_logits = policy(states)
            returns = get_returns(rewards)
            m = torch.distributions.Categorical(action_logits)
            loss = torch.mean(-m.log_prob(torch.tensor(actions)) * torch.tensor(returns))
            cum_loss += loss.item()
            loss.backward()
            optimizer.step()
        paths = []
        if epoch % LOG_FREQ == 0:
            torch.save(policy.model.state_dict(), SAVEPATH)
            torch.save(optimizer.state_dict(), SAVEPATH + '.optim')
            losses.append(cum_loss/EPIS_PER_EPOCH)
            temp_losses = np.array(losses)
            np.save('temp_policy_loss.npy', temp_losses)

losses = np.array(losses)
np.save('policy_loss.npy', losses)


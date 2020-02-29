from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import torch.nn
import torch.distributions
import random

from .policy import Policy

# CONSTANTS
N_EPOCH = 1e3
EPIS_PER_EPOCH = 5
MAX_BUF = 1e4
END_TOKEN = '<|endoftext|>'
EPS_DECAY = .9
N_ACTIONS = 5
STATE_SZ = 768
BATCH_SZ = 100
N_BATCHES = 5 
MAX_PATH = 1e4
gamma = .9
learning_rate = .01

paths = []

device = torch.device("cuda" if args.cuda else "cpu")


''' GPT 2 GENERATING CODE '''
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config()
config.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)

generated = tokenizer.encode("The Manhattan Bridge")
context = torch.tensor([generated])
past = None
sequence = tokenizer.decode(generated)


''' POLICY TRAINING CODE '''
policy = Policy()

optimizer = torch.optim.RMSprop(policy.parameters(), lr=learning_rate)

eps = .9

def get_returns(rewards):
    returns = np.zeros(len(rewards))
    for i in range(len(rewards)):
        r = rewards[i:]
        discount = [gamma ** t for t in range(len(r))]
        returns[i] = np.array(r) @ np.array(gamma)
    return returns

with open(fname, 'w') as outf:
    for epoch in range(N_EPOCH):
        for epi in range(EPIS_PER_EPOCH):
            gen_episode(outf, epi)
        policy.train()
        for path in paths:
            optimizer.zero_grad()
            states = path['state']
            actions = path['action']
            rewards = path['reward']
            action_logits = policy(states)
            returns = get_returns(rewards)
            m = Categorical(action_logits)
            loss = -m.log_prob(actions) * returns
            loss.backward()
            optimizer.step()
        paths = []

def gen_episode(outf, epi):
    import pdb;pdb.set_trace()
    with torch.no_grad(): #TODO replace corpus.dictionary.word2idx
        path = {}
        policy.eval() # freeze
        generated = tokenizer.encode("The Manhattan Bridge")
        context = torch.tensor([generated])
        past = None
        # generate trajectory for episode
        while sequence[-1] != END_TOKEN:
            output = model(context, past=past)
            logits = output[0]
            hiddens = output[2]
            state = hiddens[-1][:,-1,:]
            action_logits = policy(state)
            rewards, idx = torch.topk(logits[:,-1,:], k=5, dim=-1)
            m = Categorial(action_logits)
            action = m.sample()
            reward = torch.softmax(rewards)[action]
            token = idx[action]
            generated += [token.tolist()]
            context = token.unsqueeze(0)
            sequence = tokenizer.decode(generated)
            path['state'].append(state)
            path['action'].append(action)
            path['reward'].append(reward)
            sep = ' ' if sequence[-1] != END_TOKEN else '\n'
            outf.write(sequence + sep)
        paths.append(path)

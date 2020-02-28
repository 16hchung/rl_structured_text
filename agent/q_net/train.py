import torch
import random

from lm.word_language_model import data
from .model import QNet

# CONSTANTS
N_EPOCH = 1e3
EPIS_PER_EPOCH = 5
MAX_BUF = 1e4
END_TOKEN = '<eos>'
EPS_DECAY = .9
N_ACTIONS = 5
STATE_SZ = 200
BATCH_SZ = 100
N_BATCHES = 5 

loss = something #TODO

curr_buff_idx = -1
state_buffer  = torch.zeros(MAX_BUF, STATE_SZ) 
action_buffer = torch.zeros(MAX_BUF, 1, dtype=torch.int) 
reward_buffer = torch.zeros(MAX_BUF, 1) 

device = torch.device("cuda" if args.cuda else "cpu")

# LM THINGS
corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

with open(args.checkpoint, 'rb') as f:
    lm = torch.load(f).to(device)
lm.eval()

qnet = QNet()
def batch_iter():
    last_idx = min(curr_buff_idx + 1, MAX_BUF)
    last_state = state_buffer[-1,:]
    next_state_buffer[:-1,:] = state_buffer[1:,:]
    next_state_buffer[0,:] = last_state
    list_zipped = list(zip(state_buffer[:last_idx, :], action_buffer[:last_idx, :], reward_buffer[:last_idx, :]))
    np.random.shuffle(list_zipped)
    states, actions, rewards = zip(*list_zipped)
    n_batches = min(math.floor(curr_buff_idx / BATCH_SZ), N_BATCHES)
    for i in range(n_batches):
        yield states[i*BATCH_SZ:(i+1)*BATCH_SZ]), actions[i*BATCH_SZ:(i+1)*BATCH_SZ]), rewards[i*BATCH_SZ:(i+1)*BATCH_SZ])

eps = .9
possible_actions = torch.tensor(list(range(N_ACTIONS)))
with open(fname, 'w') as outf:
    for epoch in range(N_EPOCH):
        for epi in range(EPIS_PER_EPOCH):
            gen_episode(outf)
        qnet.train()
        for states, actions, rewards in batch_iter():
            Q(states, actions)
            calculate target using reward buffer
            update loss
            log loss
            cache generated text
            eps = eps * EPS_DECAY

def gen_episode(outf):
    with torch.no_grad():
        qnet.eval() # freeze
        curr_token = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
        curr_hidden = model.init_hidden(1)
        
        # generate trajectory for episode
        while curr_token.item() != corpus.dictionary.word2idx[END_TOKEN]:
            output, hidden = lm(curr_token, hidden)
            import pdb;pdb.set_trace()
            word_weights = output.squeeze().div(1).exp().cpu()
            ww_sorted, ww_idcs = torch.sort(word_weights, dim=-1, descending=True)
            action = random.randint(0, N_ACTIONS-1)
            if random.random() > eps:
                q_pred = qnet(hidden.repeat(N_ACTIONS,1), possible_actions)
                action = torch.argmax(q_pred)
            word_idx = ww_idcs[action]
            curr_token.fill_(word_idx)
            reward = torch.softmax(ww_sorted)[action]
            # update buffer
            curr_buff_idx += 1
            state_buffer[curr_buff_idx % MAX_BUF,:] = hidden
            reward_buffer[curr_buff_idx % MAX_BUF,:] = reward
            action_buffer[curr_buff_idx % MAX_BUF,:] = action
            sep = ' ' if word_idx != corpus.dictionary.word2idx[END_TOKEN] else '\n'

            outf.write(corpus.dictionary.idx2word[word_idx] + sep)

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import random

from .model import QNet

# CONSTANTS
N_EPOCH = 1e3
EPIS_PER_EPOCH = 5
MAX_BUF = 1e4
END_TOKEN = '<|endoftext|>'
EPS_DECAY = .9
N_ACTIONS = 5
STATE_SZ = 200
BATCH_SZ = 100
N_BATCHES = 5 

curr_buff_idx = -1
state_buffer  = torch.zeros(MAX_BUF, STATE_SZ) 
action_buffer = torch.zeros(MAX_BUF, 1, dtype=torch.int) 
reward_buffer = torch.zeros(MAX_BUF, 1) 

device = torch.device("cuda" if args.cuda else "cpu")


''' GPT 2 GENERATING CODE '''
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config()
import pdb;pdb.set_trace()
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

''' QNET TRAINING CODE '''
qnet = QNet()
def batch_iter():
    last_idx = min(curr_buff_idx + 1, MAX_BUF)
    next_state_buffer = torch.roll(state_buffer, 1, 0)
    list_zipped = list(zip(state_buffer[:last_idx, :], action_buffer[:last_idx, :], reward_buffer[:last_idx, :], next_state_buffer[:last_idx, :]))
    np.random.shuffle(list_zipped)
    states, actions, rewards, next_states = zip(*list_zipped)
    n_batches = min(math.floor(curr_buff_idx / BATCH_SZ), N_BATCHES)
    for i in range(n_batches):
        start_idx = i*BATCH_SZ
        end_idx = (i+1)*BATCH_SZ 
        yield states[start_idx:end_idx], next_states[start_idx:end_idx], actions[start_idx:end_idx], rewards[start_idx:end_idx]

eps = .9
possible_actions = torch.tensor(list(range(N_ACTIONS)))

with open(fname, 'w') as outf:
    for epoch in range(N_EPOCH):
        for epi in range(EPIS_PER_EPOCH):
            gen_episode(outf)
        qnet.train()
        for states, actions, rewards in batch_iter():
            q_hat = Q(states, actions)
            target = rewards + torch.max(Q(next_states, possible_actions))
            calculate target using reward buffer
            update loss
            log loss
            cache generated text
            eps = eps * EPS_DECAY

def gen_episode(outf):
    with torch.no_grad():
        qnet.eval() # freeze
        #curr_token = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
        #curr_hidden = model.init_hidden(1)
        generated = tokenizer.encode("The Manhattan Bridge")# TODO generate random token
        context = torch.tensor([generated])
        past = None
       
        # generate trajectory for episode
        while sequence[-1] != END_TOKEN:
            output = model(context, past=past)
            logits = output[0]
            hiddens = output[2]
            q_state = hiddens[-1][:,-1,:]
            action = random.randint(0, N_ACTIONS-1)
            if random.random() > eps:
                q_pred = qnet(hidden.repeat(N_ACTIONS, 1), possible_actions)
                action = torch.argmax(q_pred)
            rewards, idx = torch.topk(logits[:,-1,:], k=5,dim=-1)
            token = idx[action]
            reward = torch.softmax(rewards)[action]
            generated += [token.tolist()]
            context = token.unsqueeze(0)
            sequence = tokenizer.decode(generated)
            curr_buff_idx += 1
            state_buffer[curr_buff_idx % MAX_BUF,:] = hidden
            reward_buffer[curr_buff_idx % MAX_BUF,:] = reward
            action_buffer[curr_buff_idx % MAX_BUF,:] = action
            sep = ' ' if sequence[-1] != END_TOKEN else '\n'
            outf.write(sequence + sep)

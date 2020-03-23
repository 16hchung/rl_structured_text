from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, BertForNextSentencePrediction, BertTokenizer, BertConfig
from copy import deepcopy
from utils import rand_gen_first_token, evaluator, prompt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import random
import math
from tqdm import tqdm

#from .model import QNet
from temp_qnet import TempQNet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpt_as_policy', action='store_true')
parser.add_argument('--r_prob_scaler', default=1.)#20.)
parser.add_argument('--r_tgt_word_scaler', default=0.)#1.)
parser.add_argument('--r_simscore_scaler', default=0.)#.5)
parser.add_argument('--r_seq_scaler', default=0.)#.5)
parser.add_argument('--r_no_repeat_scaler', default=0.)#5.)
cmd_args = parser.parse_args()

# CONSTANTS
SENT_END_TOKENS = ['.', '?', '!'] #, '\n']
N_EPOCH = 300
EXPLOR_EPOCHS = 50
EPIS_PER_EPOCH = 5
MAX_BUF = int(1e4)
MIN_LENGTH = 100
END_TOKEN = '<|endoftext|>'
EPS_DECAY = .97
N_ACTIONS = 3
STATE_SZ = 768
BATCH_SZ = 100
N_BATCHES = 5 
N_TOKEN_OPTIONS = 100
TGT_UPDATE_FREQ = 10
DISCOUNT = .99
MAX_LENGTH = 993 #1024-- GENERATED TEXT LIMITED TO 1024 BY THE GPT2 POSITIONAL ENCODINGS STRUCTURE
SAVEPATH = 'temp_q_network.bin'
LOG_FREQ = 2
fname = 'temp_q_generated_txt.'
TEMPS = [.7, .8, .9, 1., 1.1, 1.2, 1.3]

target_words = torch.load('target_words.pt').tolist()

curr_buff_idx = -1
# initialized in train(...) (don't take up memory unless we need to)
state_buffer  = None 
action_buffer = None 
reward_buffer = None 


def batch_iter(device):
    global curr_buff_idx, state_buffer, reward_buffer, action_buffer
    last_idx = min(curr_buff_idx + 1, MAX_BUF)
    next_state_buffer = torch.roll(state_buffer, 1, 0)
    list_zipped = list(zip(state_buffer[:last_idx, :], next_state_buffer[:last_idx,:], action_buffer[:last_idx, :], reward_buffer[:last_idx, :]))
    np.random.shuffle(list_zipped)
    states, next_states, actions, rewards = [torch.stack(t) for t in zip(*list_zipped)]
    n_batches = min(math.floor(curr_buff_idx / BATCH_SZ), N_BATCHES)
    for i in range(n_batches):
        start_idx = i*BATCH_SZ
        end_idx = (i+1)*BATCH_SZ 
        yield states[start_idx:end_idx, :], next_states[start_idx:end_idx, :], actions[start_idx:end_idx, :], rewards[start_idx:end_idx, :]

def train(outf):
    global curr_buff_idx, state_buffer, reward_buffer, action_buffer

    eps = .5

    tokenizer, model, qnet, target_qnet, device = init_models()

    state_buffer  = torch.zeros((MAX_BUF, STATE_SZ), device=device)
    action_buffer = torch.zeros((MAX_BUF, 1), dtype=torch.int, device=device) 
    reward_buffer = torch.zeros((MAX_BUF, 1), device=device)

    bert_model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    optimizer = torch.optim.Adam(qnet.parameters(), lr=0.001)
    optimizer.load_state_dict(torch.load(SAVEPATH + '.optim'))
    criterion = torch.nn.MSELoss()
    losses = []
    cum_rewards = []
    for epoch in range(N_EPOCH):
        # generate episodes
        with torch.no_grad(): 
            for epi in range(EPIS_PER_EPOCH):
                cum_reward = gen_episode(model, qnet, tokenizer, outf, eps, device)
                cum_rewards.append(cum_reward)
        # backprop into q network
        qnet.train()
        cum_loss = 0
        for states, next_states, actions, rewards in batch_iter(device):
            qnet.zero_grad()
            q_hat = qnet(states).gather(-1,actions.long())
            # generate possible_actions: shape = (batch size, n_actions)
            #possible_actions = torch.tensor(
            #    [[[i] for i in range(N_ACTIONS)] for _ in range(len(actions))], 
            #    device=device, dtype=torch.float
            #)
            ##possible_actions = torch.tensor(list(range(N_ACTIONS)) * len(actions)) 
            #target = rewards + DISCOUNT * torch.max(
            #    torch.cat(
            #        [target_qnet(next_states, possible_actions[:,i]) for i in range(N_ACTIONS)], 1
            #    ), dim=1, keepdims=True
            #).values
            target = rewards + DISCOUNT * torch.max(target_qnet(next_states), dim=1, keepdims=True).values

            loss = criterion(q_hat, target)
            loss.backward()
            cum_loss += loss.item()
            optimizer.step()
            #log loss
            eps = eps * EPS_DECAY
            #if epoch > EXPLOR_EPOCHS:
            #    eps = eps * EPS_DECAY
        # update target network
        if epoch % TGT_UPDATE_FREQ == -1:
            target_qnet.load_state_dict(qnet.state_dict())
        # save everything/print every once in a while
        if epoch % LOG_FREQ == 0:
            torch.save(qnet.state_dict(), SAVEPATH)
            torch.save(optimizer.state_dict(), SAVEPATH + '.optim')
            losses.append(cum_loss/EPIS_PER_EPOCH)
            print('cumulative loss: {}'.format(cum_loss/EPIS_PER_EPOCH))
            np.save('temp_q_loss.npy', np.array(losses))
            np.save('temp_q_reward.npy', np.array(cum_rewards))
        print('epoch done')

def init_models(pretrained_path = None):
    device = torch.device("cuda")# if args.cuda else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = GPT2Config()
    config.output_hidden_states = True
    model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)

    model.to(device)
    model.cuda()

    qnet = TempQNet() 
    qnet.to(device)
    qnet.cuda()
    qnet.load_state_dict(torch.load(SAVEPATH))
    #if pretrained_path != None: qnet.load_state_dict(torch.load(pretrained_path))

    target_qnet = TempQNet() 
    target_qnet.to(device)
    target_qnet.cuda()
    target_qnet.load_state_dict(qnet.state_dict())
    target_qnet.eval()

    return tokenizer, model, qnet, target_qnet, device

def generate(outf, pretrained_path, n_essays=100):
    with torch.no_grad():
        tokenizer, model, qnet, target_qnet, device = init_models(pretrained_path)

        rewards = []
        for _ in tqdm(range(n_essays)):
            r = gen_episode(model, qnet, tokenizer, outf, 0, device, False)
            rewards.append(r)
        rewards = np.array(rewards)
        np.save('eval_rewards.npy', rewards)

def gen_episode(model, qnet, tokenizer, outf, eps, device, save_buffs=True):
    global curr_buff_idx, state_buffer, reward_buffer, action_buffer
    qnet.eval() # freeze
    model.eval()

    possible_actions = torch.unsqueeze(torch.tensor(list(range(N_ACTIONS)), device=device, dtype=torch.float), 1)
    #generated = tokenizer.encode("The Manhattan Bridge")
    generated, past, init_state = rand_gen_first_token(model, tokenizer, device)
    context = torch.tensor([generated], device=device)
    sequence = tokenizer.decode(generated)
    eps_reward = 0.0
    eos_encode = tokenizer.encode([END_TOKEN])[0]
    length = 0
    prev_sent = None
    cur_sent = [generated[0]]
    last_token = 0
    curr_temp_idx = 3
    # generate trajectory for episode
    while generated[-1] != eos_encode:
        logits, past, hiddens = model(context, past=past)
        q_state = hiddens[-1][...,-1,:]
        action = random.randint(0, N_ACTIONS-1)
        if random.random() > eps:
            q_pred = qnet(q_state)
            action = torch.argmax(q_pred).item()
        # change temperature
        if action == 0:
            curr_temp_idx = max(0, curr_temp_idx-1)
        elif action == 2:
            curr_temp_idx = min(len(TEMPS)-1, curr_temp_idx+1)
        temperature = TEMPS[curr_temp_idx]
        probs, idx = torch.topk(logits[...,-1,:].squeeze(0), k=N_TOKEN_OPTIONS, dim=-1)
        probs = torch.softmax(probs/temperature, -1)
        m = torch.distributions.Categorical(probs)
        token_idx = m.sample().item()
        token = idx[..., token_idx]
        generated += [token.tolist()]
        context = token.unsqueeze(0)
        sequence = tokenizer.decode(generated)
        r_prob = probs[token_idx].item()
        r_tgt = 0. #1. if token in target_words else 0.
        r_simscore = 0. # TODO
        reward = cmd_args.r_prob_scaler*r_prob + cmd_args.r_tgt_word_scaler*r_tgt + cmd_args.r_simscore_scaler*r_simscore
        eps_reward += reward
        if save_buffs:
            curr_buff_idx += 1
            state_buffer[curr_buff_idx % MAX_BUF,:] = q_state
            reward_buffer[curr_buff_idx % MAX_BUF,:] = reward
            action_buffer[curr_buff_idx % MAX_BUF,:] = action
        if len(generated) + len(prompt.split()) >= 1000:
            if save_buffs: reward_buffer[curr_buff_idx % MAX_BUF,:] = -1000
            sequence += END_TOKEN
            eps_reward -= 1000
            break
        #sep = " " if sequence[-1] != END_TOKEN else "\n"
    if len(generated) < MIN_LENGTH:
        if save_buffs: reward_buffer[curr_buff_idx % MAX_BUF, :] = -1000
        eps_reward -= 1000
    outf.write(sequence)
    print('finished ep')
    if random.random() < .05:
        print(sequence)
        print(action_buffer[(curr_buff_idx-100)%MAX_BUF:curr_buff_idx%MAX_BUF,:].squeeze(-1))
    return eps_reward #/ len(generated)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--pretrained_path', type=str, default=None)
    args = parser.parse_args()
    
    if args.eval:
        with open('tempqnet_eval_generated.txt', 'w') as f:
            generate(f, args.pretrained_path)
    else:
        with open(fname, 'w') as outf:
            train(outf)


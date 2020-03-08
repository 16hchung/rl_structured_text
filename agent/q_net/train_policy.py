from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, BertForNextSentencePrediction, BertTokenizer
from utils import rand_gen_first_token, get_essay_samples, evaluator
import torch
import torch.nn
import torch.distributions
import random
import numpy as np
from policy import Policy
import pickle
import math
import h5py
import argparse
from sequential import bert_seq
parser = argparse.ArgumentParser()
parser.add_argument('--gpt_as_policy', action='store_true')
parser.add_argument('--r_prob_scaler', default=1.)
parser.add_argument('--r_tgt_word_scaler', default=1.)
parser.add_argument('--r_simscore_scaler', default=.5)
parser.add_argument('--r_seq_scaler', default=.5)
cmd_args = parser.parse_args()

# CONSTANTS
SENT_END_TOKENS = ['.', '?', '!'] #, '\n']
N_EPOCH = int(200)
LOG_FREQ = 2
EPIS_PER_EPOCH = 5
MAX_BUF = 1e4
MIN_LENGTH = 100
END_TOKEN = '<|endoftext|>'
EPS_DECAY = .9
N_ACTIONS = 5
STATE_SZ = 768
BATCH_SZ = 100
N_BATCHES = 5 
MAX_PATH = 1e4
SAVEPATH = 'dummy.bin'
gamma = .99
learning_rate = .01
fname = 'dummy.txt'
MAX_LENGTH = 994 #1024
FILEPATH = 'dummy'
paths = []

#device = torch.device("cuda" if args.cuda else "cpu")
target_words = torch.load('target_words.pt').tolist()

''' GPT 2 GENERATING CODE '''
device = torch.device('cuda')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config()
config.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
model.eval()
model.to(device)
model.cuda()
bert_model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

''' POLICY TRAINING CODE '''
if cmd_args.gpt_as_policy:
    policy = GPT2LMHeadModel.from_pretrained('gpt2', config = config)
else:
    policy = Policy()
policy.to(device)
policy.cuda()

SAMPLES = get_essay_samples()
#policy.model.load_state_dict(torch.load(SAVEPATH))

if cmd_args.gpt_as_policy:
    optimizer = torch.optim.RMSprop(policy.lm_head.parameters(), lr=learning_rate)
else:
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=learning_rate)

eps = .99

def get_returns(rewards):
    returns = np.zeros(len(rewards))
    cum_rewards = np.sum(rewards)
    sent_length = len(rewards)
    for i in range(len(rewards)):
        r = rewards[i:]
        discount = [gamma ** t for t in range(len(r))]
        returns[i] = np.array(r) @ np.array(discount)
    return returns, cum_rewards, sent_length

def gen_episode(outf, epi, f, epoch, data):
    d = data.create_group('eps' + str(epi))
    reward_sum = 0.0
    dt = h5py.string_dtype(encoding='ascii') 
    d.create_dataset('state', (MAX_LENGTH+1,), dtype=dt)
    d.create_dataset('emb_state', (MAX_LENGTH+1,768), dtype='f')
    d.create_dataset('action', (MAX_LENGTH+1,), dtype='i')
    d.create_dataset('prob', (MAX_LENGTH+1,), dtype='f')
    d.create_dataset('scaled_reward_prob', (MAX_LENGTH+1,), dtype='f')
    d.create_dataset('scaled_reward_target', (MAX_LENGTH+1,), dtype='f')
    d.create_dataset('scaled_reward_sim', (MAX_LENGTH+1,), dtype='f')
    d.create_dataset('combined_reward', (MAX_LENGTH+1,), dtype='f')
    d.create_dataset('final_length', (1,), dtype='f')
    d.create_dataset('final_reward', (1,), dtype='f')
    with torch.no_grad():
        model.eval()
        path = {}
        path['state'] = []
        path['action'] = []
        path['reward'] = []
        policy.eval() # freeze
        generated, past, init_state = rand_gen_first_token(model, tokenizer, device=device) #None)
        context = torch.tensor([generated], device=device)
        sequence = tokenizer.decode(generated)
        f['epoch' + str(epoch)]['eps' + str(epi)]['state'][0] = tokenizer.decode(generated[-1])
        f['epoch' + str(epoch)]['eps' + str(epi)]['emb_state'][0] = init_state.cpu()
        length = 0
        prev_sent = None
        cur_sent = [generated[0]]
        # generate trajectory for episode
        while generated[-1] != tokenizer.encode([END_TOKEN])[0] and length < MAX_LENGTH-1:
            logits, past, hiddens = model(context, past=past)
            #logits = output[0]
            #past = output[1]
            #hiddens = output[2] 
            if len(hiddens[-1].shape) > 2:
                state = hiddens[-1][:,-1,:]
            else:
                state = hiddens[-1]
            if cmd_args.gpt_as_policy:
                logits, _,_ = policy(state)
                probs = torch.softmax(logits, -1)
                m = torch.distributions.Categorical(probs)
                action = m.sample().item()
                r_prob = probs[action].item()
                token = action
            else:
                action_logits = policy(state)
                probs, idx = torch.topk(logits[...,-1,:], k=5, dim=-1)
                m = torch.distributions.Categorical(action_logits)
                action = m.sample().item()
                probs = probs.squeeze(0)
                r_prob = torch.softmax(probs, -1)[action].item()
                idx = idx.squeeze(0)
                token = idx[action]
            generated += [token.tolist()]
            cur_sent += [token.tolist()]
            length += 1
            context = token.unsqueeze(0)
            sequence = tokenizer.decode(generated)
            r_tgt = 1. if token in target_words else 0.
            r_simscore = math.sqrt(abs((init_state[0] @ state[0]).item()))/768.0
            sim_reward = cmd_args.r_simscore_scaler * r_simscore
            tgt_reward = cmd_args.r_tgt_word_scaler * r_tgt
            prob_reward = cmd_args.r_prob_scaler * r_prob
            r_seq = 0.0
            if tokenizer.decode(generated[-1]) in SENT_END_TOKENS:
                cur_sent = tokenizer.decode(cur_sent)
                if prev_sent is not None:
                    r_seq = bert_seq(device, bert_model, bert_tokenizer, prev_sent, cur_sent)
                prev_sent = cur_sent
                cur_sent = []
            seq_reward = cmd_args.r_seq_scaler * r_seq
            reward = prob_reward + tgt_reward + sim_reward + seq_reward
            f['epoch' + str(epoch)]['eps' + str(epi)]['state'][length+1] = tokenizer.decode(generated[-1])
            f['epoch' + str(epoch)]['eps' + str(epi)]['emb_state'][length+1] = state.cpu()
            f['epoch' + str(epoch)]['eps' + str(epi)]['scaled_reward_prob'][length] = prob_reward
            f['epoch' + str(epoch)]['eps' + str(epi)]['scaled_reward_target'][length] = tgt_reward
            f['epoch' + str(epoch)]['eps' + str(epi)]['scaled_reward_sim'][length] = sim_reward
            f['epoch' + str(epoch)]['eps' + str(epi)]['combined_reward'][length] = reward
            f['epoch' + str(epoch)]['eps' + str(epi)]['prob'][length] = r_prob
            f['epoch' + str(epoch)]['eps' + str(epi)]['action'][length] = action
            path['state'].append(state)
            path['action'].append(action)
            path['reward'].append(np.array(reward))
        outf.write(sequence)
        r_score = evaluator(SAMPLES, model, tokenizer, sequence, device)
        f['epoch' + str(epoch)]['eps' + str(epi)]['final_length'][0] = length 
        f['epoch' + str(epoch)]['eps' + str(epi)]['final_reward'][0] = reward_sum
        if length < MIN_LENGTH:
            path['reward'][-1] -= 200
            f['epoch' + str(epoch)]['eps' + str(epi)]['final_reward'][0] -= 200
            f['epoch' + str(epoch)]['eps' + str(epi)]['combined_reward'][-1] -= 200
        if length >= MAX_LENGTH:
            outf.write(END_TOKEN)
            path['reward'][-1] -= 100
            f['epoch' + str(epoch)]['eps' + str(epi)]['final_reward'][0] -= 100
            f['epoch' + str(epoch)]['eps' + str(epi)]['combined_reward'][-1] -= 100
        paths.append(path)

losses = []
length_arr = []
rewards_arr = []
with open(fname, 'w') as outf:
    f = h5py.File(FILEPATH + '.hdf5', 'w')
    for epoch in range(N_EPOCH):
        data = f.create_group('epoch' + str(epoch))
        for epi in range(EPIS_PER_EPOCH):
            gen_episode(outf, epi, f, epoch, data)
            #outf.write('\n')
            #outf.write('\n')
        policy.train()
        cum_loss = 0
        # train the network after every epoch (= 5 eps per epoch)
        for path in paths:
            optimizer.zero_grad()
            states = path['state']
            actions = path['action']
            rewards = path['reward']
            states = torch.cat(states).cuda()
            rewards = np.array(rewards)
            action_logits = policy(states)
            returns, cum_rewards, sent_length = get_returns(rewards)
            rewards_arr.append(cum_rewards)
            length_arr.append(sent_length)
            m = torch.distributions.Categorical(action_logits)
            loss = torch.mean(-m.log_prob(torch.tensor(actions, device=device)) * torch.tensor(returns, device=device))
            cum_loss += loss.item()
            loss.backward()
            optimizer.step()
        paths = []
        if epoch % LOG_FREQ == 0:
            torch.save(policy.model.state_dict(), SAVEPATH)
            torch.save(optimizer.state_dict(), SAVEPATH + '.optim')
            losses.append(cum_loss/EPIS_PER_EPOCH)
            temp_losses = np.array(losses)
            COMMON_NAME = 'dummy_'
            np.save(COMMON_NAME + 'training_loss.npy', temp_losses)
            rewards_np = np.array(rewards_arr)
            np.save(COMMON_NAME + 'rewards.npy', rewards_np)
            length_np = np.array(length_arr)
            np.save(COMMON_NAME + 'length.npy', length_np)
        print('Finished epoch!' + str(epoch))
    f.close()
    print('DONE')

losses = np.array(losses)
np.save('policy_loss.npy', losses)


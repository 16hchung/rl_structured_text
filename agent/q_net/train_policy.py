from copy import deepcopy
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, BertForNextSentencePrediction, BertTokenizer, BertConfig
from utils import rand_gen_first_token, get_essay_samples, evaluator, prompt
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
parser.add_argument('--r_prob_scaler', default=10.)#20.)
parser.add_argument('--r_tgt_word_scaler', default=0.)#1.)
parser.add_argument('--r_simscore_scaler', default=0.)#.5)
parser.add_argument('--r_seq_scaler', default=0.)#.5)
parser.add_argument('--r_no_repeat_scaler', default=15.)#5.)
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
SAVEPATH = 'GPTasPG_combined_r_network.bin' if cmd_args.gpt_as_policy else 'PG_combined_r_network.bin'
gamma = .99
learning_rate = .01
fname = 'GPTasPG_combined_r.txt' if cmd_args.gpt_as_policy else 'PG_combined_r.txt'
MAX_LENGTH = 993 #1024-- GENERATED TEXT LIMITED TO 1024 BY THE GPT2 POSITIONAL ENCODINGS STRUCTURE
FILEPATH = 'GPTasPG_combined_r' if cmd_args.gpt_as_policy else 'PG_combined_r'
paths = []
#temperature = 2.
#temp_decay = .999
temperature = 1.3
temp_decay = .99985

#device = torch.device("cuda" if args.cuda else "cpu")
target_words = torch.load('target_words.pt').squeeze(-1).tolist()

''' GPT 2 GENERATING CODE '''
device = torch.device('cuda')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config()
config.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
model.eval()
model.to(device)
model.cuda()
#bert_config = BertConfig()
#bert_config.max_position_embeddings = 1024
## MAX POSITIONAL ENCODINGS OF BERT LIMITS THE SENTENCE LENGTH TO 512 TOKENS
bert_model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
''' POLICY TRAINING CODE '''
if cmd_args.gpt_as_policy:
    policy = model
    model_lm_state_dict = deepcopy(model.lm_head.state_dict())
    for param in policy.transformer.parameters():
        param.requires_grad = False
    for param in policy.lm_head.parameters():
        param.requires_grad = True
    policy.load_state_dict(torch.load('GPTasPG_combined_r_network.bin'))
else:
    policy = Policy()
policy.to(device)
policy.cuda()

#SAMPLES = get_essay_samples()
#policy.model.load_state_dict(torch.load(SAVEPATH))

if cmd_args.gpt_as_policy:
    optimizer = torch.optim.AdamW(policy.lm_head.parameters(), lr=.0001)
    optimizer.load_state_dict(torch.load('GPTasPG_combined_r_network.bin.optim'))
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

def repeats_ngram(generated):
    gen_len = len(generated)
    max_n = min(gen_len / 2, 10)
    for n in range(1, int(max_n)+1):
        if generated[gen_len-n:] == generated[gen_len-2*n:gen_len-n]:
            return True
    return False

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
        if cmd_args.gpt_as_policy:
            pol_lm_state_dict = deepcopy(model.lm_head.state_dict())
            model.lm_head.load_state_dict(model_lm_state_dict)
        generated, past, init_state = rand_gen_first_token(model, tokenizer, device=device) #None)
        if cmd_args.gpt_as_policy:
            policy.lm_head.load_state_dict(pol_lm_state_dict)
            prompt_encode = tokenizer.encode(prompt) 
            ctxt = torch.tensor([prompt_encode]).cuda()
            _, action_past, _ = policy(ctxt, past=None)
        context = torch.tensor([generated], device=device)
        sequence = tokenizer.decode(generated)
        f['epoch' + str(epoch)]['eps' + str(epi)]['state'][0] = tokenizer.decode(generated[-1])
        f['epoch' + str(epoch)]['eps' + str(epi)]['emb_state'][0] = init_state.cpu()
        length = 0
        prev_sent = None
        cur_sent = [generated[0]]
        last_token = 0
        # generate trajectory for episode
        while generated[-1] != tokenizer.encode([END_TOKEN])[0] and length < MAX_LENGTH-1:
            #logits = output[0]
            #past = output[1]
            #hiddens = output[2] 
            if cmd_args.gpt_as_policy:
                pol_lm_state_dict = deepcopy(model.lm_head.state_dict())
                model.lm_head.load_state_dict(model_lm_state_dict)
            logits, past, hiddens = model(context, past=past)
            if cmd_args.gpt_as_policy:
                policy.lm_head.load_state_dict(pol_lm_state_dict)
            if len(hiddens[-1].shape) > 2:
                state = hiddens[-1][:,-1,:]
            else:
                state = hiddens[-1]
            if cmd_args.gpt_as_policy:
                # sample action from policy net output
                action_logits, action_past, _ = policy(context, past=action_past)
                action_probs = torch.softmax(action_logits / temperature, -1).squeeze(0).squeeze(0)
                m = torch.distributions.Categorical(action_probs)
                action = m.sample().item()
                # get reward from model net
                probs = torch.softmax(logits, -1).squeeze(0).squeeze(0)
                r_prob = probs[action].item()
                token = torch.tensor(action, device=device)
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
            r_no_repeat = -1. if repeats_ngram(generated) else 0.
            last_token = token.item()
            r_tgt = 1. if last_token in target_words else 0.
            r_simscore = math.sqrt(abs((init_state[0] @ state[0]).item()))/768.0
            sim_reward = cmd_args.r_simscore_scaler * r_simscore
            no_repeat_reward = cmd_args.r_no_repeat_scaler * r_no_repeat
            tgt_reward = cmd_args.r_tgt_word_scaler * r_tgt
            prob_reward = cmd_args.r_prob_scaler * r_prob
            r_seq = 0.0
            if False and tokenizer.decode(generated[-1]) in SENT_END_TOKENS:
                cur_sent = tokenizer.decode(cur_sent[:511])[:511]
                if prev_sent is not None: 
                    r_seq = bert_seq(device, bert_model, bert_tokenizer, prev_sent, cur_sent)
                prev_sent = cur_sent[:511]
                cur_sent = []
            seq_reward = cmd_args.r_seq_scaler * r_seq
            reward = prob_reward + tgt_reward + sim_reward + seq_reward + no_repeat_reward
            try:
                f['epoch' + str(epoch)]['eps' + str(epi)]['state'][length+1] = tokenizer.decode(generated[-1])
                f['epoch' + str(epoch)]['eps' + str(epi)]['emb_state'][length+1] = state.cpu()
                f['epoch' + str(epoch)]['eps' + str(epi)]['scaled_reward_prob'][length] = prob_reward
                f['epoch' + str(epoch)]['eps' + str(epi)]['scaled_reward_target'][length] = tgt_reward
                f['epoch' + str(epoch)]['eps' + str(epi)]['scaled_reward_sim'][length] = sim_reward
                f['epoch' + str(epoch)]['eps' + str(epi)]['combined_reward'][length] = reward
                f['epoch' + str(epoch)]['eps' + str(epi)]['prob'][length] = r_prob
                f['epoch' + str(epoch)]['eps' + str(epi)]['action'][length] = action
            except:
                print('hfd5 error')
            if not cmd_args.gpt_as_policy:
                path['state'].append(state)
            path['action'].append(action)
            path['reward'].append(np.array(reward))
        outf.write(sequence)
        if cmd_args.gpt_as_policy:
            path['state'] = generated
        #r_score = evaluator(SAMPLES, model, tokenizer, sequence, device)
        f['epoch' + str(epoch)]['eps' + str(epi)]['final_length'][0] = length 
        f['epoch' + str(epoch)]['eps' + str(epi)]['final_reward'][0] = reward_sum
        if length < MIN_LENGTH:
            path['reward'][-1] -= 1000
            f['epoch' + str(epoch)]['eps' + str(epi)]['final_reward'][0] -= 1000
            f['epoch' + str(epoch)]['eps' + str(epi)]['combined_reward'][-1] -= 1000
        if length >= MAX_LENGTH:
            outf.write(END_TOKEN)
            path['reward'][-1] -= 1000
            f['epoch' + str(epoch)]['eps' + str(epi)]['final_reward'][0] -= 1000
            f['epoch' + str(epoch)]['eps' + str(epi)]['combined_reward'][-1] -= 1000
        paths.append(path)
        print('ep done')

losses = []
length_arr = []
rewards_arr = []
max_cum_loss = 0
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
            actions = path['action']
            rewards = path['reward']
            rewards = np.array(rewards)
            if cmd_args.gpt_as_policy:
                prompt_encode = tokenizer.encode(prompt) 
                ctxt = torch.tensor([prompt_encode + path['state']]).cuda()
                action_logits = policy(ctxt, past=None)[0][0,len(prompt_encode)+1:,:]
                action_logits = torch.softmax(action_logits / temperature, -1)
            else:
                states = path['state']
                states = torch.cat(states).cuda()
                action_logits = policy(states)
            returns, cum_rewards, sent_length = get_returns(rewards)
            rewards_arr.append(cum_rewards)
            length_arr.append(sent_length)
            m = torch.distributions.Categorical(action_logits)
            loss = torch.mean(-m.log_prob(torch.tensor(actions, device=device)) * torch.tensor(returns, device=device))
            cum_loss += loss.item()
            loss.backward()
            optimizer.step()
        if cmd_args.gpt_as_policy:
            last_sent = tokenizer.decode(paths[-1]['state'])
        paths = []
        temperature = max(1., temperature*temp_decay)
        if epoch % LOG_FREQ == 0:
            if max_cum_loss < cum_loss:
                torch.save(policy.state_dict() if cmd_args.gpt_as_policy else policy.model.state_dict(), SAVEPATH)
                torch.save(optimizer.state_dict(), SAVEPATH + '.optim')
                max_cum_loss = cum_loss
            losses.append(cum_loss/EPIS_PER_EPOCH)
            temp_losses = np.array(losses)
            loss_file = 'GPTasPG_combined_r_training_loss.npy' if cmd_args.gpt_as_policy else 'PG_combined_r_training_loss.npy'
            np.save(loss_file, temp_losses)
            rewards_np = np.array(rewards_arr)
            rewards_file = 'GPTasPG_combined_r_rewards.npy' if cmd_args.gpt_as_policy else 'PG_combined_r_rewards.npy'
            np.save(rewards_file, rewards_np)
            length_np = np.array(length_arr)
            length_file = 'GPTasPG_combined_r_length.npy' if cmd_args.gpt_as_policy else 'PG_combined_r_length.npy'
            np.save(length_file, length_np)
            if cmd_args.gpt_as_policy:
                print(last_sent)
        print('Finished epoch!' + str(epoch))
    f.close()
    print('DONE')

losses = np.array(losses)
np.save('GPTaspolicy_loss.npy' if cmd_args.gpt_as_policy else 'policy_loss.npy', losses)


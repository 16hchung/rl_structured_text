from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from tqdm import tqdm

from .model import QNet

# CONSTANTS
N_EPOCH = int(1e3)
EPIS_PER_EPOCH = 5
MAX_BUF = int(1e4)
END_TOKEN = '<|endoftext|>'
EPS_DECAY = .9
N_ACTIONS = 5
STATE_SZ = 768
BATCH_SZ = 100
N_BATCHES = 5 
TGT_UPDATE_FREQ = 10
DISCOUNT = .9
SAVEPATH = 'q_network.bin'
LOG_FREQ = 2
fname = 'q_generated_txt.'
prompt = 'Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive?'

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

    state_buffer  = torch.zeros((MAX_BUF, STATE_SZ), device=device)
    action_buffer = torch.zeros((MAX_BUF, 1), dtype=torch.int, device=device) 
    reward_buffer = torch.zeros((MAX_BUF, 1), device=device)

    eps = .9

    tokenizer, model, qnet, target_qnet, device = init_models()

    optimizer = torch.optim.Adam(qnet.parameters(), lr=0.05)
    #optimizer.load_state_dict(torch.load(SAVEPATH + '.optim'))
    criterion = torch.nn.MSELoss()
    cum_loss = 0
    losses = []
    for epoch in range(N_EPOCH):
        # generate episodes
        with torch.no_grad(): 
            for epi in range(EPIS_PER_EPOCH):
                gen_episode(model, qnet, tokenizer, outf, eps, device)
        # backprop into q network
        qnet.train()
        for states, next_states, actions, rewards in batch_iter(device):
            qnet.zero_grad()
            q_hat = qnet(states, actions.float())
            # generate possible_actions: shape = (batch size, n_actions)
            possible_actions = torch.tensor(
                [[[i] for i in range(N_ACTIONS)] for _ in range(len(actions))], 
                device=device, dtype=torch.float
            )
            #possible_actions = torch.tensor(list(range(N_ACTIONS)) * len(actions)) 
            target = rewards + DISCOUNT * torch.max(
                torch.cat(
                    [target_qnet(next_states, possible_actions[:,i]) for i in range(N_ACTIONS)], 1
                ), dim=1, keepdims=True
            ).values

            loss = criterion(q_hat, target)
            loss.backward()
            cum_loss += loss.item()
            optimizer.step()
            #log loss
            #cache generated text
            eps = eps * EPS_DECAY
        # update target network
        if epoch % TGT_UPDATE_FREQ == -1:
            target_qnet.load_state_dict(qnet.state_dict())
        # save everything/print every once in a while
        if epoch % LOG_FREQ == 0:
            torch.save(qnet.state_dict(), SAVEPATH)
            torch.save(optimizer.state_dict(), SAVEPATH + '.optim')
            losses.append(cum_loss/EPIS_PER_EPOCH)
            print('cumulative loss: {}'.format(cum_loss/EPIS_PER_EPOCH))
            np.save('q_loss.npy', np.array(losses))

def init_models(pretrained_path = None):
    device = torch.device("cuda")# if args.cuda else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = GPT2Config()
    config.output_hidden_states = True
    model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)

    model.to(device)
    model.cuda()

    qnet = QNet() 
    qnet.to(device)
    qnet.cuda()
    if pretrained_path != None: qnet.load_state_dict(torch.load(pretrained_path))

    target_qnet = QNet() 
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

    possible_actions = torch.unsqueeze(torch.tensor(list(range(N_ACTIONS)), device=device, dtype=torch.float), 1)
    #generated = tokenizer.encode("The Manhattan Bridge")
    generated, past = rand_gen_first_token(model, tokenizer, device)
    context = torch.tensor([generated], device=device)
    sequence = tokenizer.decode(generated)
    eps_reward = 0.0
    eos_encode = tokenizer.encode([END_TOKEN])[0]
    # generate trajectory for episode
    while generated[-1] != eos_encode:
        logits, past, hiddens = model(context, past=past)
        q_state = hiddens[-1][...,-1,:]
        action = random.randint(0, N_ACTIONS-1)
        if random.random() > eps:
            q_pred = qnet(q_state.repeat(N_ACTIONS, 1).cuda(), possible_actions)
            action = torch.argmax(q_pred)
        rewards, idx = torch.topk(logits[...,-1,:], k=N_ACTIONS,dim=-1)
        token = idx[..., action]
        reward = torch.softmax(rewards[0], 0)[action]
        eps_reward += reward
        generated += token.tolist()
        context = token.unsqueeze(0)
        sequence = tokenizer.decode(generated)
        if save_buffs:
            curr_buff_idx += 1
            state_buffer[curr_buff_idx % MAX_BUF,:] = q_state
            reward_buffer[curr_buff_idx % MAX_BUF,:] = reward
            action_buffer[curr_buff_idx % MAX_BUF,:] = action
        if len(generated) + len(prompt.split()) >= 1000:
            if save_buffs: reward_buffer[curr_buff_idx % MAX_BUF,:] = -100
            eps_reward -= 100
            break
        #sep = " " if sequence[-1] != END_TOKEN else "\n"
    outf.write(sequence)
    return eps_reward / len(generated)

def rand_gen_first_token(model, tokenizer, device):
    okay_tokens = [1002 , 314  , 1471 , 2141 , 383  , 843  , 1867 , 632  , 921  , 1148 , 4231 , 4362 , 770  , 554  , 
                   1320 , 1374 , 775  , 1400 , 4162 , 4222 , 366  , 10928, 317  , 10358, 3363 , 8192 , 1649 , 1680 , 
                   1114 , 1081 , 8314 , 2312 , 1318 , 1675 , 23998, 1119 , 3406 , 3226 , 4418 , 3914 , 7731 , 1892 , 
                   3894 , 3244 , 4377 , 2011 , 6350 , 8673 , 4619 , 1406 , 3412 , 10347, 2773 , 2293 , 887  , 1881 , 
                   28416, 352  , 2750 , 6952 , 1629 , 8013 , 1550 , 2329 , 1703 , 5845 , 4784 , 4650 , 6674 , 4042 , 
                   1052 , 3274 , 1439 , 2561 , 362  , 4900 , 16263, 5338 , 15176, 25110, 7875 , 2893 , 9022 , 15323, 
                   3423 , 3954 , 2080 , 18948, 4149 , 2094 , 4091 , 4525 , 4380 , 513  , 14026, 6521 , 28417, 43048, 
                   3574 , 6930 , 23631, 1355 , 18578, 9170 , 11399, 2102 , 1770 , 8447 , 2735 , 5514 , 367  , 838  , 
                   767  , 16981, 3582 , 642  , 10127, 6914 , 9506 , 7214 , 604  , 8989 , 8920 , 12642]
    # TODO do we wanna include this as a state/action pair? or just as random init?
    #      ie word we generate here is s_0
    # start state is beginning of seq token (actually same as end of seq token)
    generated = tokenizer.encode(prompt)
    context = torch.tensor([generated], device=device)
    # do one iteration but not greedy to generate random token
    logits, past, _= model(context, past=None)
    token = None
    while token not in okay_tokens:
        topk = torch.topk(F.softmax(logits[...,-1,:], 1),150)
        token_idx = torch.multinomial(topk.values, 1).item()
        token = topk.indices[0,token_idx].item()
    return [token], past

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--pretrained_path', type=str, default=None)
    args = parser.parse_args()
    
    if args.eval:
        with open('eval_generated.txt', 'w') as f:
            generate(f, args.pretrained_path)
    else:
        with open(fname, 'w') as outf:
            train(outf)


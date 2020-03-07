from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

# GLOBAL
prompt = 'Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive?'


#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#config = GPT2Config()
#config.output_hidden_states = True

def rand_gen_first_token(model, tokenizer, device):
    #prompt = 'Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive?'
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
    context = torch.tensor([generated]) #, device=torch.device('cpu'))
    # do one iteration but not greedy to generate random token
    model.to(device)
    model.cuda()
    logits, past, hiddens= model(context.to(device), past=None)
    token = None
    if len(hiddens[-1].shape) > 2:
        state = hiddens[-1][:,-1,:]
    else:
        state = hiddens[-1]
    while token not in okay_tokens:
        topk = torch.topk(F.softmax(logits[...,-1,:], 1),150)
        token_idx = torch.multinomial(topk.values, 1).item()
        token = topk.indices[0,token_idx].item()
    return [token], past, state

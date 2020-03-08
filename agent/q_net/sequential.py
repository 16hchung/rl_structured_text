from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer

def bert_seq(device, model, tokenizer, s1, s2):
    encoded = tokenizer.encode_plus(s1, text_pair=s2, return_tensors='pt')
    seq_relationship_logits = model(**encoded)[0]
    probs = softmax(seq_relationship_logits, dim=1)
    return probs[0,0].item()

def main():
    model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    seq_A = 'I like cookies !'
    seq_B = 'Do you like them ?'
    probs = bert_seq(model, tokenizer, seq_A, seq_B)
    print(probs)



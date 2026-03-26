"""
Project: Generative-Pretraining-Transformer-Scratchpad
Author: Jixuan Yao
Date: March 2026

LEARNING SOURCE:
This project is a dedicated study of the "Neural Networks: Zero to Hero" series 
by Andrej Karpathy. 
Course Reference: https://github.com/karpathy/ng-video-lecture

PURPOSE:
- Hand-coded reproduction: Every line is manually implemented while following 
  the lecture to internalize the transformer architecture.
- Detailed Lab Notes

TECHNICAL FOCUS:
- Building a character-level language model from the ground up.
- Step-by-step evolution: From simple Bigram statistics to Multi-head Self-Attention.
- Data: Tiny Shakespeare dataset (Vocab size: 65).
"""


import os
import urllib.request
import torch

# data preparation
data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
save_path = "input.txt"

if not os.path.exists(save_path): 
    print("Downlooading Datasets...")
    urllib.request.urlretrieve(data_url, save_path)
    print(f'Download sucessfully, saved in {save_path}')
else: print(f"Already exist{save_path}")

# -----------------------------------------------------------------------------------------------------------------------
# Read File

with open(save_path, 'r', encoding='utf-8' ) as f:
    text = f.read()

print('length of dataset:',len(text))

print(text[:1000])

# all the unique characters 
chars = sorted(list(set(text))) 
vocab_size = len(chars)
print(' '.join(chars))
print(vocab_size)

# create a mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)} # 字符到数字的字典
itos = {i:ch for i,ch in enumerate(chars)}# 数字到字符的字典

encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join(itos[i] for i in l)

print(encode('hi, there'))
print(decode(encode('hi, there'))) 

# encode entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.type) # data tensor 
# print(data[:1000])

# -----------------------------------------------------------------------------------------------------------------------
# split data into train and val sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
print(train_data[: block_size+1])

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[:t]
    print(f'when input is {context}, the target is {target}')

# -----------------------------------------------------------------------------------------------------------------------
print('-----------------------------------------------------------------------------------------------------------------------')
torch.manual_seed(1337)
batch_size = 4 #how much of independent sequences are proceed in parallel
block_size = 8 # maximum context length for predictions?

def get_batch(split):
    data = train_data if split == "train" else val_data 
    # torch. randint(high, size) 生成一个张量，里面是随机数，范围是[0, high). len(data)-block_size 防止越界 ↓
    ix = torch.randint(len(data) - block_size, (batch_size,))   # batch size of random off sets into the training sets
    x = torch.stack([data[i:i+block_size] for i in ix]) # 将one dimensional tensor stack up into one tensor as rows
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape) 
print(xb)
print('targets:')
print(yb.shape)
print(yb)

# only process one batch:
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]  # xb[b][0:t+1]  第b行的[0:t+1]
        target = xb[b,t]
        print(f'When input is {context}, the target is: {target}')

# 输出：
# inputs:
# torch.Size([4, 8])
# tensor([[24, 43, 58,  5, 57,  1, 46, 43],     # 整个张量是一个batch, each of row is a chunk of training set
#         [44, 53, 56,  1, 58, 46, 39, 58],
#         [52, 58,  1, 58, 46, 39, 58,  1],
#         [25, 17, 27, 10,  0, 21,  1, 54]])
# targets:
# torch.Size([4, 8])
# tensor([[43, 58,  5, 57,  1, 46, 43, 39],
#         [53, 56,  1, 58, 46, 39, 58,  1],
#         [58,  1, 58, 46, 39, 58,  1, 46],
#         [17, 27, 10,  0, 21,  1, 54, 39]])
# When input is tensor([24]), the target is: 24
# When input is tensor([24, 43]), the target is: 43
# When input is tensor([24, 43, 58]), the target is: 58
# When input is tensor([24, 43, 58,  5]), the target is: 5
# When input is tensor([24, 43, 58,  5, 57]), the target is: 57
# When input is tensor([24, 43, 58,  5, 57,  1]), the target is: 1
# When input is tensor([24, 43, 58,  5, 57,  1, 46]), the target is: 46
# When input is tensor([24, 43, 58,  5, 57,  1, 46, 43]), the target is: 43
# When input is tensor([44]), the target is: 44
# When input is tensor([44, 53]), the target is: 53
# When input is tensor([44, 53, 56]), the target is: 56
# When input is tensor([44, 53, 56,  1]), the target is: 1
# .... Four Batches


# -----------------------------------------------------------------------------------------------------------------------
print('-----------------------------------------------------------------------------------------------------------------------')

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# Bigram Model : 只看当前预测下一个
print(xb)  # Input to the Transformer

class BigramLanguageModel(nn.Module): 
    def __init__ (self, vocab_size):
        super().__init__()
        # 创建一个trainnable lookup tableb 并自动随机初始化， 储存每个token对应下一个的概率
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # 输出logits, P（y|x) = softmax(logits)

    def forward(self, idx, targets):
        # 一次取多个logits, idx 是Batch( 总共有多少个样本), Time（每个样本的长度） Tensor of integers
        # idx example: idx = [
                            # [0, 1, 2, 3],
                            # [4, 5, 6, 7]
                            # ]

        # logits shape: (B, T, C)
        logits = (self.token_embedding_table(idx))  # embedding table 是一个类而不是列表或字典，所以要用'()’
        B, T , C = logits.shape
        logits = logits.view(B*T, C) #将三维压缩成2维，才能计算cross_entropy
        targets = targets.view(B*T) #同理，二维压缩成一维
        loss = F.cross_entropy(logits, targets)

        return logits, loss
    
m = BigramLanguageModel(vocab_size) # vocab_size 65
logits, loss = m(xb, yb)
print(xb.shape) # [4,8] 4个独立句子，8个Token
print(yb.shape) # [4,8]
print(logits.shape) # logits 若未经过'压缩'：[4,8，65] 对4*8 = 32 个字符中的每一个都查出了65个数值，它们代表了下一个字符是vocabulary 1-65 个中每一个的可能性
                 # 这里是压缩后的所以是[32,65]
print(loss)






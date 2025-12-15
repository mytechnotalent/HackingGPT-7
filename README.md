![image](https://github.com/mytechnotalent/HackingGPT-7/blob/main/HackingGPT.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# HackingGPT
## Part 7: Building a Complete GPT from Scratch

Part 7 brings everything together to build a complete GPT language model. We implement the full transformer architecture with:

- **Multi-head self-attention** (the core mechanism)
- **Feed-forward networks** with GELU activation
- **Residual connections** and **layer normalization**
- **AdamW optimizer** with **cosine learning rate scheduling**
- **Gradient clipping** and **dropout** for stable training
- **Early stopping with patience** to prevent overfitting and save compute time
- **Model saving/loading** for inference

By the end, you'll have a working character-level language model trained on Sherlock Holmes!

#### Author: [Kevin Thomas](mailto:ket189@pitt.edu)

<br>

## Part 6 [HERE](https://github.com/mytechnotalent/HackingGPT-6)

<br><br>

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
```


## Step 1: Load and Inspect the Data

Our GPT learns from a text file (Sherlock Holmes stories). Key things we'll explore:

- **Dataset size**: ~580K characters - enough to learn English patterns
- **Vocabulary**: All unique characters (letters, numbers, punctuation)
- **Character-level tokenization**: Each character = one token (simple but effective)

We'll create mappings between characters and integers so the model can process text as numbers.


```python
# set random seed for reproducibility
torch.manual_seed(1337)
```


**Output:**
```
<torch._C.Generator at 0x10bf3e510>
```


```python
# read the input text file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```


```python
# display basic statistics about the text
print('dataset statistics')
print()
print(f'total characters: {len(text):,}')
print(f'first 200 characters:')
print(text[:200])
```


**Output:**
```
dataset statistics

total characters: 581,565
first 200 characters:
﻿The Project Gutenberg eBook of The Adventures of Sherlock Holmes
    
This ebook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no

```


```python
# get all unique characters in the text (our vocabulary)
chars = sorted(list(set(text)))
chars
```


**Output:**
```
['\n',
 ' ',
 '!',
 '#',
 '$',
 '%',
 '&',
 '(',
 ')',
 '*',
 ',',
 '-',
 '.',
 '/',
 '0',
 '1',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '9',
 ':',
 ';',
 '?',
 'A',
 'B',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'J',
 'K',
 'L',
 'M',
 'N',
 'O',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'U',
 'V',
 'W',
 'X',
 'Y',
 'Z',
 '[',
 ']',
 '_',
 'a',
 'b',
 'c',
 'd',
 'e',
 'f',
 'g',
 'h',
 'i',
 'j',
 'k',
 'l',
 'm',
 'n',
 'o',
 'p',
 'q',
 'r',
 's',
 't',
 'u',
 'v',
 'w',
 'x',
 'y',
 'z',
 '£',
 '½',
 'à',
 'â',
 'æ',
 'è',
 'é',
 'œ',
 '—',
 '‘',
 '’',
 '“',
 '”',
 '•',
 '™',
 '\ufeff']
```


```python
# vocab_size: number of unique characters
vocab_size = len(chars)
vocab_size
```


**Output:**
```
98
```


```python
# display the vocabulary
print('vocabulary')
print()
print(f'vocab_size = {vocab_size}')
print(f'characters: {repr("".join(chars))}')
```


**Output:**
```
vocabulary

vocab_size = 98
characters: '\n !#$%&()*,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz£½àâæèéœ—‘’“”•™\ufeff'

```


```python
# create character to integer mapping
# stoi = "string to integer"
stoi = {ch: i for i, ch in enumerate(chars)}
stoi
```


**Output:**
```
{'\n': 0,
 ' ': 1,
 '!': 2,
 '#': 3,
 '$': 4,
 '%': 5,
 '&': 6,
 '(': 7,
 ')': 8,
 '*': 9,
 ',': 10,
 '-': 11,
 '.': 12,
 '/': 13,
 '0': 14,
 '1': 15,
 '2': 16,
 '3': 17,
 '4': 18,
 '5': 19,
 '6': 20,
 '7': 21,
 '8': 22,
 '9': 23,
 ':': 24,
 ';': 25,
 '?': 26,
 'A': 27,
 'B': 28,
 'C': 29,
 'D': 30,
 'E': 31,
 'F': 32,
 'G': 33,
 'H': 34,
 'I': 35,
 'J': 36,
 'K': 37,
 'L': 38,
 'M': 39,
 'N': 40,
 'O': 41,
 'P': 42,
 'Q': 43,
 'R': 44,
 'S': 45,
 'T': 46,
 'U': 47,
 'V': 48,
 'W': 49,
 'X': 50,
 'Y': 51,
 'Z': 52,
 '[': 53,
 ']': 54,
 '_': 55,
 'a': 56,
 'b': 57,
 'c': 58,
 'd': 59,
 'e': 60,
 'f': 61,
 'g': 62,
 'h': 63,
 'i': 64,
 'j': 65,
 'k': 66,
 'l': 67,
 'm': 68,
 'n': 69,
 'o': 70,
 'p': 71,
 'q': 72,
 'r': 73,
 's': 74,
 't': 75,
 'u': 76,
 'v': 77,
 'w': 78,
 'x': 79,
 'y': 80,
 'z': 81,
 '£': 82,
 '½': 83,
 'à': 84,
 'â': 85,
 'æ': 86,
 'è': 87,
 'é': 88,
 'œ': 89,
 '—': 90,
 '‘': 91,
 '’': 92,
 '“': 93,
 '”': 94,
 '•': 95,
 '™': 96,
 '\ufeff': 97}
```


```python
# create integer to character mapping
# itos = "integer to string"
itos = {i: ch for i, ch in enumerate(chars)}
itos
```


**Output:**
```
{0: '\n',
 1: ' ',
 2: '!',
 3: '#',
 4: '$',
 5: '%',
 6: '&',
 7: '(',
 8: ')',
 9: '*',
 10: ',',
 11: '-',
 12: '.',
 13: '/',
 14: '0',
 15: '1',
 16: '2',
 17: '3',
 18: '4',
 19: '5',
 20: '6',
 21: '7',
 22: '8',
 23: '9',
 24: ':',
 25: ';',
 26: '?',
 27: 'A',
 28: 'B',
 29: 'C',
 30: 'D',
 31: 'E',
 32: 'F',
 33: 'G',
 34: 'H',
 35: 'I',
 36: 'J',
 37: 'K',
 38: 'L',
 39: 'M',
 40: 'N',
 41: 'O',
 42: 'P',
 43: 'Q',
 44: 'R',
 45: 'S',
 46: 'T',
 47: 'U',
 48: 'V',
 49: 'W',
 50: 'X',
 51: 'Y',
 52: 'Z',
 53: '[',
 54: ']',
 55: '_',
 56: 'a',
 57: 'b',
 58: 'c',
 59: 'd',
 60: 'e',
 61: 'f',
 62: 'g',
 63: 'h',
 64: 'i',
 65: 'j',
 66: 'k',
 67: 'l',
 68: 'm',
 69: 'n',
 70: 'o',
 71: 'p',
 72: 'q',
 73: 'r',
 74: 's',
 75: 't',
 76: 'u',
 77: 'v',
 78: 'w',
 79: 'x',
 80: 'y',
 81: 'z',
 82: '£',
 83: '½',
 84: 'à',
 85: 'â',
 86: 'æ',
 87: 'è',
 88: 'é',
 89: 'œ',
 90: '—',
 91: '‘',
 92: '’',
 93: '“',
 94: '”',
 95: '•',
 96: '™',
 97: '\ufeff'}
```


```python
# encode function: convert string to list of integers
def encode(s):
    '''
    Convert a string to a list of integers.
    
    Args:
        s: input string to encode
        
    Returns:
        list of integers representing each character
    '''
    return [stoi[c] for c in s]
```


```python
# decode function: convert list of integers back to string
def decode(l):
    '''
    Convert a list of integers back to a string.
    
    Args:
        l: list of integers to decode
        
    Returns:
        string representation of the integers
    '''
    return ''.join([itos[i] for i in l])
```


```python
# test encode and decode
print('testing encode and decode')
print()
test_string = 'hello'
encoded = encode(test_string)
decoded = decode(encoded)
print(f'original: {repr(test_string)}')
print(f'encoded: {encoded}')
print(f'decoded: {repr(decoded)}')
print(f'match: {test_string == decoded}')
```


**Output:**
```
testing encode and decode

original: 'hello'
encoded: [63, 60, 67, 67, 70]
decoded: 'hello'
match: True

```


```python
# encode the entire text into a tensor
data = torch.tensor(encode(text), dtype=torch.long)
data
```


**Output:**
```
tensor([97, 46, 63,  ...,  0,  0,  0])
```


```python
# display data tensor info
print('encoded data tensor')
print()
print(f'shape: {data.shape}')
print(f'dtype: {data.dtype}')
print(f'first 100 tokens: {data[:100].tolist()}')
```


**Output:**
```
encoded data tensor

shape: torch.Size([581565])
dtype: torch.int64
first 100 tokens: [97, 46, 63, 60, 1, 42, 73, 70, 65, 60, 58, 75, 1, 33, 76, 75, 60, 69, 57, 60, 73, 62, 1, 60, 28, 70, 70, 66, 1, 70, 61, 1, 46, 63, 60, 1, 27, 59, 77, 60, 69, 75, 76, 73, 60, 74, 1, 70, 61, 1, 45, 63, 60, 73, 67, 70, 58, 66, 1, 34, 70, 67, 68, 60, 74, 0, 1, 1, 1, 1, 0, 46, 63, 64, 74, 1, 60, 57, 70, 70, 66, 1, 64, 74, 1, 61, 70, 73, 1, 75, 63, 60, 1, 76, 74, 60, 1, 70, 61, 1]

```


```python
# split data into training and validation sets
# 90% for training, 10% for validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```


```python
# display split information
print('train/validation split')
print()
print(f'total tokens: {len(data):,}')
print(f'train tokens: {len(train_data):,} ({100*len(train_data)/len(data):.1f}%)')
print(f'val tokens: {len(val_data):,} ({100*len(val_data)/len(data):.1f}%)')
```


**Output:**
```
train/validation split

total tokens: 581,565
train tokens: 523,408 (90.0%)
val tokens: 58,157 (10.0%)

```


## Step 2: Define All Hyperparameters

Hyperparameters control the model's architecture and training behavior. Unlike model weights (which are learned), hyperparameters are set by us before training.

**Architecture hyperparameters** determine model size:
- `n_embd`, `n_head`, `n_layer` → bigger = more capacity but slower

**Training hyperparameters** control how the model learns:
- `batch_size`, `learning_rate`, `max_iters` → affect convergence speed and quality

**Regularization hyperparameters** prevent overfitting:
- `dropout` → randomly drops connections during training to improve generalization

**Early stopping hyperparameters** save compute time:
- `patience` → number of evaluations without improvement before stopping training early


```python
# batch_size: how many independent sequences to process in parallel
# larger = faster training but more memory
batch_size = 64
batch_size
```


**Output:**
```
64
```


```python
# block_size: maximum context length for predictions
# this is how many tokens the model can "see" when making a prediction
block_size = 256
block_size
```


**Output:**
```
256
```


```python
# max_iters: total number of training iterations
max_iters = 10000
max_iters
```


**Output:**
```
10000
```


```python
# eval_interval: how often to evaluate loss on train/val sets
eval_interval = 500
eval_interval
```


**Output:**
```
500
```


```python
# learning_rate: step size for optimizer
# too high = unstable, too low = slow learning
learning_rate = 5e-4
learning_rate
```


**Output:**
```
0.0005
```


```python
# device: use GPU if available for faster training
# priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    # NVIDIA GPU (Windows/Linux)
    device = 'cuda'
elif torch.backends.mps.is_available():
    # Apple Silicon GPU (Mac M1/M2/M3)
    device = 'mps'
else:
    # fallback to CPU
    device = 'cpu'
device
```


**Output:**
```
'mps'
```


```python
# eval_iters: how many batches to average over when estimating loss
eval_iters = 200
eval_iters
```


**Output:**
```
200
```


```python
# n_embd: embedding dimension (size of token representations)
# larger = more expressive but more parameters
n_embd = 384
n_embd
```


**Output:**
```
384
```


```python
# n_head: number of attention heads in multi-head attention
# each head learns different patterns
n_head = 6
n_head
```


**Output:**
```
6
```


```python
# n_layer: number of transformer blocks stacked
# deeper = more complex patterns but harder to train
n_layer = 6
n_layer
```


**Output:**
```
6
```


```python
# dropout: probability of dropping units during training
# helps prevent overfitting
dropout = 0.35
dropout
```


**Output:**
```
0.35
```


```python
# patience: number of evaluations to wait for improvement before stopping
# if validation loss doesn't improve for this many checks, training stops early
# with eval_interval=500 and patience=5, stops after 2500 steps of no improvement
patience = 5
patience
```


**Output:**
```
5
```


```python
# display all hyperparameters together
print('hyperparameter summary')
print()
print(f'batch_size = {batch_size}')
print(f'block_size = {block_size}')
print(f'max_iters = {max_iters}')
print(f'eval_interval = {eval_interval}')
print(f'learning_rate = {learning_rate}')
print(f'eval_iters = {eval_iters}')
print(f'n_embd = {n_embd}')
print(f'n_head = {n_head}')
print(f'n_layer = {n_layer}')
print(f'dropout = {dropout}')
print(f'patience = {patience}')
print()
print('derived values')
print(f'head_size = n_embd // n_head = {n_embd} // {n_head} = {n_embd // n_head}')
print()
print('device information')
print(f'device = {device}')
if device == 'cuda':
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   CUDA version: {torch.version.cuda}')
elif device == 'mps':
    print('   GPU: Apple Silicon (MPS)')
else:
    print('   using CPU (no GPU acceleration)')
```


**Output:**
```
hyperparameter summary

batch_size = 64
block_size = 256
max_iters = 10000
eval_interval = 500
learning_rate = 0.0005
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.35
patience = 5

derived values
head_size = n_embd // n_head = 384 // 6 = 64

device information
device = mps
   GPU: Apple Silicon (MPS)

```


## Step 3: Create Batch Generator

Training on one example at a time is inefficient. We batch multiple sequences together for the following.
- **GPU parallelism**: Process many sequences simultaneously
- **Stable gradients**: Averaging over a batch reduces noise
- **Faster training**: More data processed per forward/backward pass

Each batch contains `batch_size` independent sequences of length `block_size`.


```python
# get_batch: generate a batch of training examples
def get_batch(split):
    '''
    Generate a batch of input-target pairs for training or validation.
    
    Args:
        split: 'train' or 'val' to select which dataset to use
        
    Returns:
        x: input tensor of shape (batch_size, block_size)
        y: target tensor of shape (batch_size, block_size)
    '''
    # select the appropriate dataset
    data = train_data if split == 'train' else val_data
    
    # generate random starting positions for each sequence in the batch
    # we need room for block_size tokens, so max start is len(data) - block_size
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # stack the sequences into batches
    # x[i] = data[ix[i] : ix[i] + block_size]
    x = torch.stack([data[i:i+block_size] for i in ix])
    
    # y is shifted by 1: y[i] = data[ix[i] + 1 : ix[i] + block_size + 1]
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # move to device (GPU if available)
    x, y = x.to(device), y.to(device)
    
    # return the input-target pair
    return x, y
```


```python
# test get_batch
xb, yb = get_batch('train')
print('testing get_batch')
print()
print(f'x shape: {xb.shape}')
print(f'y shape: {yb.shape}')
print()
print('first sequence in batch')
print(f'x[0]: {xb[0].tolist()[:10]}... (first 10 tokens)')
print(f'y[0]: {yb[0].tolist()[:10]}... (first 10 tokens)')
print()
print('note: y is x shifted by 1 position')
print(f'x[0][1] = {xb[0][1].item()} should equal y[0][0] = {yb[0][0].item()}')
```


**Output:**
```
testing get_batch

x shape: torch.Size([64, 256])
y shape: torch.Size([64, 256])

first sequence in batch
x[0]: [1, 59, 64, 73, 60, 58, 75, 70, 73, 74]... (first 10 tokens)
y[0]: [59, 64, 73, 60, 58, 75, 70, 73, 74, 1]... (first 10 tokens)

note: y is x shifted by 1 position
x[0][1] = 59 should equal y[0][0] = 59

```


## Step 4: Loss Estimation Function

We need a reliable way to measure model performance during training. Rather than using a single batch (which is noisy), we average loss over many batches for a stable estimate.

**Why average over multiple batches?**
- Single batch loss fluctuates randomly
- Averaging gives smoother, more reliable signal
- We evaluate on both train and validation sets to detect overfitting


```python
# estimate_loss: average loss over multiple batches
@torch.no_grad()
def estimate_loss(model):
    '''
    Estimate loss by averaging over eval_iters batches.
    
    Uses @torch.no_grad() decorator to disable gradient computation
    since we are only evaluating, not training.
    
    Args:
        model: the model to evaluate
        
    Returns:
        dict with 'train' and 'val' average losses
    '''
    # initialize output dictionary for train and val losses
    out = {}
    
    # set model to evaluation mode
    model.eval()
    
    # iterate over train and val splits
    for split in ['train', 'val']:
        # create tensor to store losses for averaging
        losses = torch.zeros(eval_iters)
        # sample eval_iters batches and compute loss for each
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        # store mean loss for this split
        out[split] = losses.mean()
    
    # set model back to training mode
    model.train()
    
    # return dictionary with train and val losses
    return out
```


## Step 5: Single Attention Head

Self-attention is the core mechanism of transformers. It allows each token to "look at" all previous tokens and decide which ones are relevant.

**The Key Concepts:**
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"  
- **Value (V)**: "What information do I provide?"

**How it works:**
1. Each token creates Q, K, V vectors via learned projections
2. Attention scores = Q @ K^T (how much does each token match?)
3. Scale by √(head_size) to keep gradients stable
4. Apply causal mask (can't attend to future tokens)
5. Softmax to get attention weights (sum to 1)
6. Weighted sum of V vectors = output

**Causal masking** is crucial for autoregressive generation: each position can only attend to earlier positions (no peeking at the future!).


```python
# Head: one head of self-attention
class Head(nn.Module):
    '''
    Single head of self-attention.
    
    Computes scaled dot-product attention:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(head_size)) @ V
    
    Args:
        head_size: dimension of queries, keys, and values
    
    Attributes:
        key: linear projection for keys (what I contain)
        query: linear projection for queries (what I'm looking for)
        value: linear projection for values (what I'll give)
        tril: lower triangular mask for causal attention
        dropout: dropout layer for regularization
    '''
    
    def __init__(self, head_size):
        '''
        Initialize the attention head.
        
        Args:
            head_size: dimension of the attention head
        '''
        super().__init__()
        
        # key projection: (n_embd) -> (head_size)
        # "what information do I contain?"
        self.key = nn.Linear(n_embd, head_size, bias=False)
        
        # query projection: (n_embd) -> (head_size)
        # "what information am I looking for?"
        self.query = nn.Linear(n_embd, head_size, bias=False)
        
        # value projection: (n_embd) -> (head_size)
        # "what information will I provide if attended to?"
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # register_buffer: not a parameter, but should be saved with model
        # this is the causal mask (lower triangular matrix)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        # type hint for tril buffer
        self.tril: torch.Tensor
        
        # dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        '''
        Forward pass of the attention head.
        
        Args:
            x: input tensor of shape (B, T, C)
               B = batch size
               T = sequence length
               C = n_embd (embedding dimension)
        
        Returns:
            out: output tensor of shape (B, T, head_size)
        '''
        # get dimensions (only T is needed for masking)
        _, T, _ = x.shape
        
        # compute keys: (B, T, C) -> (B, T, head_size)
        k = self.key(x)
        
        # compute queries: (B, T, C) -> (B, T, head_size)
        q = self.query(x)
        
        # compute attention scores (affinities)
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # scale by 1/sqrt(head_size) to prevent softmax from becoming too peaky
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        
        # apply causal mask: positions can only attend to previous positions
        # mask future positions with -inf so softmax gives them 0 weight
        tril = self.tril
        wei = wei.masked_fill(tril[:T, :T] == 0, float('-inf'))
        
        # apply softmax to get attention weights (probabilities)
        wei = F.softmax(wei, dim=-1)
        
        # apply dropout
        wei = self.dropout(wei)
        
        # compute values: (B, T, C) -> (B, T, head_size)
        v = self.value(x)
        
        # weighted aggregation of values
        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = wei @ v
        
        # return the attention output
        return out
```


```python
# test Head class
print('testing Head class')
print()
head_size = n_embd // n_head
print(f'head_size = n_embd // n_head = {n_embd} // {n_head} = {head_size}')
print()
test_head = Head(head_size)
print(f'key projection: {test_head.key}')
print(f'query projection: {test_head.query}')
print(f'value projection: {test_head.value}')
print()
print(f'causal mask shape: {test_head.tril.shape}')
print(f'first 4x4 of causal mask:')
tril_tensor = test_head.tril
print(tril_tensor[:4, :4])
```


**Output:**
```
testing Head class

head_size = n_embd // n_head = 384 // 6 = 64

key projection: Linear(in_features=384, out_features=64, bias=False)
query projection: Linear(in_features=384, out_features=64, bias=False)
value projection: Linear(in_features=384, out_features=64, bias=False)

causal mask shape: torch.Size([256, 256])
first 4x4 of causal mask:
tensor([[1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]])

```


## Step 6: Multi-Head Attention

A single attention head has limited expressive power. Multi-head attention runs **multiple heads in parallel**, allowing the model to attend to different patterns simultaneously.

**Why multiple heads?**
- One head might focus on nearby tokens (local patterns)
- Another might focus on syntactic relationships
- Another might track semantic meaning
- Different heads specialize for different tasks

**How it works:**
1. Run `n_head` independent attention operations in parallel
2. Concatenate all outputs along the feature dimension
3. Apply a final linear projection to mix information across heads

With `n_head=6` and `head_size=64`, we get 6 different "perspectives" on the relationships in the sequence.


```python
# MultiHeadAttention: multiple heads of self-attention in parallel
class MultiHeadAttention(nn.Module):
    '''
    Multi-head self-attention.
    
    Runs multiple attention heads in parallel and concatenates the results.
    Then projects back to the embedding dimension.
    
    The idea is that each head can learn to attend to different things:
    - One head might attend to syntax
    - Another might attend to semantics
    - Another might attend to position patterns
    
    Args:
        num_heads: number of attention heads
        head_size: dimension of each head
    
    Attributes:
        heads: ModuleList of Head modules
        proj: output projection back to n_embd
        dropout: dropout layer for regularization
    '''
    
    def __init__(self, num_heads, head_size):
        '''
        Initialize multi-head attention.
        
        Args:
            num_heads: number of parallel attention heads
            head_size: dimension of each attention head
        '''
        super().__init__()
        
        # create num_heads attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
        # output projection: (num_heads * head_size) -> (n_embd)
        # this is the Wo matrix in the paper
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        
        # dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        '''
        Forward pass of multi-head attention.
        
        Args:
            x: input tensor of shape (B, T, C)
        
        Returns:
            out: output tensor of shape (B, T, C)
        '''
        # run all heads in parallel and concatenate along the last dimension
        # each head outputs (B, T, head_size)
        # concatenation gives (B, T, num_heads * head_size)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # project back to embedding dimension and apply dropout
        out = self.dropout(self.proj(out))
        
        # return the multi-head attention output
        return out
```


```python
# test MultiHeadAttention
print('testing MultiHeadAttention')
print()
head_size = n_embd // n_head
print(f'n_head = {n_head}')
print(f'head_size = {head_size}')
print(f'n_head * head_size = {n_head * head_size} = n_embd = {n_embd}')
print()
test_mha = MultiHeadAttention(n_head, head_size)
print(f'number of heads: {len(test_mha.heads)}')
print(f'output projection: {test_mha.proj}')
print()
print('shape flow:')
print(f'   input: (B, T, {n_embd})')
print(f'   each head: (B, T, {head_size})')
print(f'   concat: (B, T, {n_head} * {head_size}) = (B, T, {n_head * head_size})')
print(f'   proj: (B, T, {n_embd})')
```


**Output:**
```
testing MultiHeadAttention

n_head = 6
head_size = 64
n_head * head_size = 384 = n_embd = 384

number of heads: 6
output projection: Linear(in_features=384, out_features=384, bias=True)

shape flow:
   input: (B, T, 384)
   each head: (B, T, 64)
   concat: (B, T, 6 * 64) = (B, T, 384)
   proj: (B, T, 384)

```


## Step 7: Feed-Forward Network

After attention, each token is processed independently by a small neural network. This is where the model does its "thinking" on each position.

**Architecture**: Linear → GELU → Linear → Dropout
- First linear layer **expands** from `n_embd` to `4 × n_embd`
- GELU activation introduces non-linearity
- Second linear layer **compresses** back to `n_embd`

**Why GELU instead of ReLU?**
- GELU (Gaussian Error Linear Unit) is smoother than ReLU
- Used in GPT-2, GPT-3, and BERT
- ReLU has a hard cutoff at 0, GELU has a smooth curve
- This helps gradients flow better during training


```python
# FeedForward: simple feed-forward network (per token)
class FeedForward(nn.Module):
    '''
    Position-wise feed-forward network.
    
    Applied to each position independently and identically.
    Expands to 4x the embedding dimension, applies GELU, then projects back.
    
    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
    
    The 4x expansion allows the model to have a larger intermediate
    representation for "thinking" before compressing back down.
    
    Args:
        n_embd: embedding dimension
    
    Attributes:
        net: Sequential network with linear, gelu, linear, dropout
    '''
    
    def __init__(self, n_embd):
        '''
        Initialize the feed-forward network.
        
        Args:
            n_embd: embedding dimension (input and output size)
        '''
        super().__init__()
        
        # build the feed-forward network
        self.net = nn.Sequential(
            # expand to 4x embedding dimension
            nn.Linear(n_embd, 4 * n_embd),
            # GELU non-linearity (smoother than ReLU, used in GPT-2/3)
            nn.GELU(),
            # project back to embedding dimension
            nn.Linear(4 * n_embd, n_embd),
            # dropout for regularization
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        '''
        Forward pass of the feed-forward network.
        
        Args:
            x: input tensor of shape (B, T, C)
        
        Returns:
            out: output tensor of shape (B, T, C)
        '''
        # apply the feed-forward network and return output
        return self.net(x)
```


```python
# test FeedForward
print('testing FeedForward')
print()
test_ffn = FeedForward(n_embd)
print(f'input dimension: {n_embd}')
print(f'hidden dimension: {4 * n_embd}')
print(f'output dimension: {n_embd}')
print()
print('network architecture:')
for i, layer in enumerate(list(test_ffn.net.children())):
    print(f'   layer {i}: {layer}')
```


**Output:**
```
testing FeedForward

input dimension: 384
hidden dimension: 1536
output dimension: 384

network architecture:
   layer 0: Linear(in_features=384, out_features=1536, bias=True)
   layer 1: GELU(approximate='none')
   layer 2: Linear(in_features=1536, out_features=384, bias=True)
   layer 3: Dropout(p=0.35, inplace=False)

```


## Step 8: Transformer Block

A transformer block is the fundamental building unit that we stack to create deep models. Each block has two main phases:

**1. Communication (Attention)**
- Tokens exchange information via multi-head attention
- Each token learns about its context

**2. Computation (Feed-Forward)**  
- Each token processes its updated representation
- Independent computation at each position

**Key Design Choices:**
- **Pre-norm architecture**: LayerNorm before (not after) each sub-layer → more stable training
- **Residual connections**: `x = x + sublayer(x)` → gradients flow easily through deep networks

The pattern repeats: `x → LayerNorm → Attention → +x → LayerNorm → FFN → +x`


```python
# Block: transformer block with communication followed by computation
class Block(nn.Module):
    '''
    Transformer block: communication (attention) followed by computation (ffn).
    
    Uses pre-norm architecture where layer norm is applied before each sub-layer.
    Residual connections allow gradients to flow directly through the network.
    
    The structure is:
        x = x + attention(ln1(x))  # communication
        x = x + ffn(ln2(x))        # computation
    
    Args:
        n_embd: embedding dimension
        n_head: number of attention heads
    
    Attributes:
        sa: self-attention module (multi-head attention)
        ffwd: feed-forward network
        ln1: first layer normalization
        ln2: second layer normalization
    '''
    
    def __init__(self, n_embd, n_head):
        '''
        Initialize the transformer block.
        
        Args:
            n_embd: embedding dimension
            n_head: number of attention heads
        '''
        super().__init__()
        
        # calculate head size: n_embd must be divisible by n_head
        head_size = n_embd // n_head
        
        # self-attention layer (communication)
        self.sa = MultiHeadAttention(n_head, head_size)
        
        # feed-forward layer (computation)
        self.ffwd = FeedForward(n_embd)
        
        # layer normalizations (pre-norm architecture)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        '''
        Forward pass of the transformer block.
        
        Args:
            x: input tensor of shape (B, T, C)
        
        Returns:
            out: output tensor of shape (B, T, C)
        '''
        # self-attention with residual connection
        # x = x + attention(ln1(x))
        x = x + self.sa(self.ln1(x))
        
        # feed-forward with residual connection
        # x = x + ffn(ln2(x))
        x = x + self.ffwd(self.ln2(x))
        
        # return the transformed output
        return x
```


```python
# test Block
print('testing Block')
print()
test_block = Block(n_embd, n_head)
print(f'n_embd = {n_embd}')
print(f'n_head = {n_head}')
print(f'head_size = {n_embd // n_head}')
print()
print('block components:')
print(f'   ln1: {test_block.ln1}')
print(f'   self-attention: MultiHeadAttention with {n_head} heads')
print(f'   ln2: {test_block.ln2}')
print(f'   ffwd: FeedForward with hidden dim {4 * n_embd}')
print()
print('data flow:')
print('   x -> ln1 -> attention -> + residual -> ln2 -> ffn -> + residual -> out')
```


**Output:**
```
testing Block

n_embd = 384
n_head = 6
head_size = 64

block components:
   ln1: LayerNorm((384,), eps=1e-05, elementwise_affine=True)
   self-attention: MultiHeadAttention with 6 heads
   ln2: LayerNorm((384,), eps=1e-05, elementwise_affine=True)
   ffwd: FeedForward with hidden dim 1536

data flow:
   x -> ln1 -> attention -> + residual -> ln2 -> ffn -> + residual -> out

```


## Step 9: Complete GPT Language Model

Now we assemble all the pieces into a complete GPT! The architecture stacks:

1. **Token Embedding**: Converts token IDs → learned vectors
2. **Position Embedding**: Adds position information (since attention has no inherent order)
3. **N Transformer Blocks**: Stack of attention + FFN (we use 6 blocks)
4. **Final LayerNorm**: Stabilizes activations before output
5. **Language Model Head**: Projects to vocabulary size for next-token prediction

The model also includes:
- **Weight initialization**: Normal distribution with std=0.02 (helps training)
- **Generate method**: Autoregressive text generation


```python
# GPTLanguageModel: the complete GPT model
class GPTLanguageModel(nn.Module):
    '''
    GPT Language Model.
    
    A decoder-only transformer that predicts the next token given previous tokens.
    
    Architecture:
        1. Token embedding: convert token indices to vectors
        2. Position embedding: add position information
        3. Transformer blocks: stack of attention + ffn blocks
        4. Final layer norm: normalize before output
        5. Language model head: project to vocabulary size for predictions
    
    Attributes:
        token_embedding_table: lookup table for token embeddings
        position_embedding_table: lookup table for position embeddings
        blocks: Sequential stack of transformer blocks
        ln_f: final layer normalization
        lm_head: linear projection to vocabulary size
    '''
    
    def __init__(self):
        '''
        Initialize the GPT language model.
        '''
        super().__init__()
        
        # token embedding table: (vocab_size, n_embd)
        # each token gets a learned vector representation
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # position embedding table: (block_size, n_embd)
        # each position gets a learned vector representation
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # stack of transformer blocks
        # Sequential applies each block in order
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        # final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)
        
        # language model head: project from n_embd to vocab_size
        # this gives us logits (unnormalized probabilities) for each token
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        '''
        Initialize weights for the model.
        
        Linear layers get normal initialization with std=0.02.
        Embedding layers get normal initialization with std=0.02.
        Biases are initialized to zero.
        
        Args:
            module: the module to initialize
        '''
        # initialize linear layers
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # initialize embedding layers
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        '''
        Forward pass of the GPT model.
        
        Args:
            idx: input token indices of shape (B, T)
            targets: target token indices of shape (B, T), optional
        
        Returns:
            logits: unnormalized predictions of shape (B, T, vocab_size)
            loss: cross-entropy loss if targets provided, else None
        '''
        # extract sequence length from input shape
        _, T = idx.shape
        
        # get token embeddings: (B, T) -> (B, T, n_embd)
        tok_emb = self.token_embedding_table(idx)
        
        # get position embeddings: (T,) -> (T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        
        # combine token and position embeddings: (B, T, n_embd)
        x = tok_emb + pos_emb
        
        # pass through transformer blocks: (B, T, n_embd)
        x = self.blocks(x)
        
        # final layer norm: (B, T, n_embd)
        x = self.ln_f(x)
        
        # project to vocabulary: (B, T, n_embd) -> (B, T, vocab_size)
        logits = self.lm_head(x)
        
        # compute loss if targets are provided
        if targets is None:
            loss = None
        else:
            # reshape for cross entropy: (B*T, vocab_size) and (B*T,)
            _, _, C_logits = logits.shape
            logits_flat = logits.view(-1, C_logits)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        # return predictions and optional loss
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''
        Generate new tokens autoregressively.
        
        Given a context, generate max_new_tokens new tokens one at a time.
        
        Args:
            idx: starting context of shape (B, T)
            max_new_tokens: number of new tokens to generate
        
        Returns:
            idx: extended sequence of shape (B, T + max_new_tokens)
        '''
        # generate one token at a time
        for _ in range(max_new_tokens):
            # crop context to block_size (model can only handle block_size tokens)
            idx_cond = idx[:, -block_size:]
            
            # get predictions using forward method
            logits, _ = self.forward(idx_cond)
            
            # focus on last time step: (B, vocab_size)
            logits = logits[:, -1, :]
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # append to running sequence: (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        # return the extended sequence
        return idx
```


## Step 10: Create and Analyze the Model

Let's instantiate our GPT and examine its structure:

- **Total parameters**: ~10.8M trainable weights
- **Device**: Automatically uses GPU (CUDA/MPS) if available, otherwise CPU
- **Memory**: Parameters stored in GPU memory for fast computation

The parameter count breakdown helps understand where the model's capacity lies.


```python
# create the model and move to device
model = GPTLanguageModel()
m = model.to(device)
print(f'model created and moved to {device}')
```


**Output:**
```
model created and moved to mps

```


```python
# count model parameters
total_params = sum(p.numel() for p in m.parameters())
print('parameter count breakdown')
print()
print(f'token embeddings: {vocab_size} x {n_embd} = {vocab_size * n_embd:,}')
print(f'position embeddings: {block_size} x {n_embd} = {block_size * n_embd:,}')
print()
print('each transformer block:')
print(f'   attention Q,K,V: 3 x {n_embd} x {n_embd // n_head} x {n_head} = {3 * n_embd * n_embd:,}')
print(f'   attention proj: {n_embd} x {n_embd} = {n_embd * n_embd:,}')
print(f'   ffn expand: {n_embd} x {4 * n_embd} = {n_embd * 4 * n_embd:,}')
print(f'   ffn contract: {4 * n_embd} x {n_embd} = {4 * n_embd * n_embd:,}')
print(f'   layer norms: 2 norms x ({n_embd} weights + {n_embd} biases) = {4 * n_embd:,}')
print()
print(f'language model head: {n_embd} x {vocab_size} = {n_embd * vocab_size:,}')
print()
print(f'total parameters: {total_params:,} ({total_params/1e6:.2f}M)')
```


**Output:**
```
parameter count breakdown

token embeddings: 98 x 384 = 37,632
position embeddings: 256 x 384 = 98,304

each transformer block:
   attention Q,K,V: 3 x 384 x 64 x 6 = 442,368
   attention proj: 384 x 384 = 147,456
   ffn expand: 384 x 1536 = 589,824
   ffn contract: 1536 x 384 = 589,824
   layer norms: 2 norms x (384 weights + 384 biases) = 1,536

language model head: 384 x 98 = 37,632

total parameters: 10,814,306 (10.81M)

```


## Step 11: Training Loop with Early Stopping

Now we train our GPT model using several advanced techniques:

### Optimizer: AdamW
- **What it is**: Adam optimizer with decoupled weight decay
- **Why we use it**: Proper L2 regularization helps prevent overfitting
- **Our setting**: `weight_decay=0.1` penalizes large weights

### Learning Rate Scheduler: Cosine Annealing
- **What it does**: Smoothly decreases learning rate following a cosine curve
- **Why we use it**: High LR early for fast progress, low LR later for fine-tuning
- **Our setting**: Decays from `1e-3` → `1e-5` over 10,000 iterations

### Gradient Clipping
- **What it does**: Limits the maximum gradient norm during backpropagation
- **Why we use it**: Prevents exploding gradients that can destabilize training
- **Our setting**: `max_norm=1.0` clips gradients if their norm exceeds 1.0

### Early Stopping with Patience
- **What it does**: Stops training when validation loss stops improving for `patience` evaluations
- **Why we use it**: Training loss can keep decreasing while validation loss rises (overfitting)
- **How patience works**: If val loss doesn't improve for `patience` consecutive checks, training stops immediately
- **Our setting**: `patience=3` with `eval_interval=500` means stop after 1,500 steps of no improvement
- **Best model restoration**: We always restore the best checkpoint, not the final (potentially overfit) model

### How Early Stopping Works Step-by-Step
1. Every `eval_interval` steps, we check validation loss
2. If val loss improved → save model weights, reset patience counter to 0
3. If val loss did NOT improve → increment patience counter by 1
4. If patience counter reaches `patience` → stop training immediately
5. After training ends → restore the saved best model weights

**Example with our settings:**
- Step 0: val loss 4.20 → best! save model, patience = 0
- Step 500: val loss 2.10 → best! save model, patience = 0  
- Step 1000: val loss 1.70 → best! save model, patience = 0
- Step 1500: val loss 1.55 → best! save model, patience = 0
- Step 2000: val loss 1.58 → worse, patience = 1/3
- Step 2500: val loss 1.60 → worse, patience = 2/3
- Step 3000: val loss 1.62 → worse, patience = 3/3 → STOP!
- Restore model from step 1500 (val loss 1.55)


```python
# create optimizer
# AdamW is Adam with proper weight decay
# weight_decay adds L2 regularization to prevent overfitting
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.15)

# create learning rate scheduler
# cosine annealing smoothly decays learning rate from max to min
# this helps the model converge better in later stages of training
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)

# display optimizer and scheduler configuration
print(f'optimizer: AdamW with learning rate {learning_rate}, weight decay 0.15')
print(f'scheduler: CosineAnnealingLR from {learning_rate} to 1e-5 over {max_iters} iterations')
```


**Output:**
```
optimizer: AdamW with learning rate 0.0005, weight decay 0.15
scheduler: CosineAnnealingLR from 0.0005 to 1e-5 over 10000 iterations

```


```python
# training loop with early stopping
# training will stop early if validation loss doesn't improve for 'patience' evaluations
print(f'training GPT for up to {max_iters} iterations')
print(f'evaluating every {eval_interval} iterations')
print(f'early stopping patience: {patience} evaluations ({patience * eval_interval} steps)')
print('=' * 60)

# track best validation loss for early stopping
# best_val_loss: the lowest validation loss seen so far (starts at infinity)
# best_model_state: a copy of the model weights at the best validation loss
# patience_counter: counts how many evaluations since last improvement (starts at 0)
# stopped_early: flag to indicate if training stopped before max_iters
best_val_loss = float('inf')
best_model_state = None
patience_counter = 0
stopped_early = False

# iterate through all training iterations
for iter in range(max_iters):
    # evaluate loss periodically (every eval_interval steps or at the last step)
    if iter % eval_interval == 0 or iter == max_iters - 1:
        # compute average loss over eval_iters batches for both train and val
        losses = estimate_loss(model)
        # get current learning rate from scheduler for logging
        current_lr = scheduler.get_last_lr()[0]
        
        # check if this is the best model so far (lowest validation loss)
        if losses['val'] < best_val_loss:
            # new best model found!
            best_val_loss = losses['val']
            # save a copy of all model weights (clone to avoid reference issues)
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            # reset patience counter since we improved
            patience_counter = 0
            print(f'step {iter:5d}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}, lr {current_lr:.2e} *best*')
        else:
            # no improvement, increment patience counter
            patience_counter += 1
            print(f'step {iter:5d}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}, lr {current_lr:.2e} (patience {patience_counter}/{patience})')
            
            # check if we've run out of patience
            if patience_counter >= patience:
                print('=' * 60)
                print(f'early stopping triggered! no improvement for {patience} evaluations')
                stopped_early = True
                break
    
    # get batch of training data
    # randomly samples batch_size sequences of length block_size
    x_batch, y_batch = get_batch('train')
    
    # forward pass: compute predictions and loss
    logits, loss = model(x_batch, y_batch)
    
    # backward pass: compute gradients
    # set_to_none=True is more efficient than zero_grad()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # gradient clipping to prevent exploding gradients
    # if the total gradient norm exceeds 1.0, scale it down
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # update weights using computed gradients
    optimizer.step()
    # decay learning rate according to cosine schedule
    scheduler.step()

# restore best model weights to avoid using an overfit model
# this ensures we use the model from the step with lowest validation loss
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print('=' * 60)
    print(f'restored best model (val loss: {best_val_loss:.4f})')

# display training completion message
print('=' * 60)
if stopped_early:
    print(f'training stopped early at step {iter}')
else:
    print('training complete (reached max_iters)')
print(f'best validation loss: {best_val_loss:.4f}')
```


**Output:**
```
training GPT for up to 10000 iterations
evaluating every 500 iterations
early stopping patience: 5 evaluations (2500 steps)
============================================================
step     0: train loss 4.5899, val loss 4.5908, lr 5.00e-04 *best*
step   500: train loss 1.8271, val loss 2.0141, lr 4.97e-04 *best*
step  1000: train loss 1.3953, val loss 1.7005, lr 4.88e-04 *best*
step  1500: train loss 1.1915, val loss 1.5746, lr 4.73e-04 *best*
step  2000: train loss 1.0829, val loss 1.5397, lr 4.53e-04 *best*
step  2500: train loss 0.9958, val loss 1.5423, lr 4.28e-04 (patience 1/5)
step  3000: train loss 0.9194, val loss 1.5349, lr 3.99e-04 *best*
step  3500: train loss 0.8458, val loss 1.5823, lr 3.66e-04 (patience 1/5)
step  4000: train loss 0.7750, val loss 1.6280, lr 3.31e-04 (patience 2/5)
step  4500: train loss 0.7059, val loss 1.6640, lr 2.93e-04 (patience 3/5)
step  5000: train loss 0.6454, val loss 1.7080, lr 2.55e-04 (patience 4/5)
step  5500: train loss 0.5815, val loss 1.7728, lr 2.17e-04 (patience 5/5)
============================================================
early stopping triggered! no improvement for 5 evaluations
============================================================
restored best model (val loss: 1.5349)
============================================================
training stopped early at step 5500
best validation loss: 1.5349

```


## Step 12: Generate Text from the Trained Model

Now we can use our trained model to generate new text! The model predicts one token at a time:

1. Feed in a starting context (or just a newline)
2. Model outputs probability distribution over next token
3. Sample from the distribution
4. Append new token to context
5. Repeat until we have enough tokens

This is called **autoregressive generation** - each new token depends on all previous tokens.


```python
# generate text from trained model
# start with a newline character (or any starting context)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print('generated text from trained GPT:')
print('=' * 60)
generated = m.generate(context, max_new_tokens=500)
print(decode(generated[0].tolist()))
print('=' * 60)
```


**Output:**
```
generated text from trained GPT:
============================================================


Friston’s vigers of the wordco insuation.”

“The Cigar Misder mornian business.”

“Burst a moment very mney.”

“Yes, sir, Mr. Holmes. It was already that I will pisend before you to
busine one which I must be so possible that never girl to be in that
I might come up in my mind work before, you have must not have from my
miles, and you chanced back too it. Was the key, that poor upon his
wrong and reway from his drive, and he returned himself in the other. His
formidation, has there in this vadi
============================================================

```


## Step 13: Save the Trained Model

Training a model takes time and resources. We save the trained model so we can:
- **Reuse it later** without retraining
- **Share it** with others
- **Deploy it** in applications

**What we save in the checkpoint:**
- `model_state_dict`: All learned weights and biases
- Architecture hyperparameters: `vocab_size`, `n_embd`, `n_head`, `n_layer`, `block_size`, `dropout`
- Tokenizer mappings: `chars`, `stoi`, `itos` (needed to encode/decode text)


```python
# save the trained model
# we save both the model state and hyperparameters needed for reconstruction

# create a checkpoint dictionary with all necessary information
checkpoint = {
    # model weights and biases
    'model_state_dict': model.state_dict(),
    
    # hyperparameters needed to reconstruct the model
    'vocab_size': vocab_size,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'block_size': block_size,
    'dropout': dropout,
    
    # tokenizer mappings (needed for encode/decode)
    'chars': chars,
    'stoi': stoi,
    'itos': itos,
}

# save to file
model_path = 'gpt_model.pt'
torch.save(checkpoint, model_path)

# display save confirmation and checkpoint contents
print(f'model saved to {model_path}')
print()
print('checkpoint contains:')
print(f'   model_state_dict: {len(checkpoint["model_state_dict"])} parameter tensors')
print(f'   vocab_size: {vocab_size}')
print(f'   n_embd: {n_embd}')
print(f'   n_head: {n_head}')
print(f'   n_layer: {n_layer}')
print(f'   block_size: {block_size}')
print(f'   dropout: {dropout}')
print(f'   chars: {len(chars)} unique characters')
print(f'   stoi: character to index mapping')
print(f'   itos: index to character mapping')
```


**Output:**
```
model saved to gpt_model.pt

checkpoint contains:
   model_state_dict: 210 parameter tensors
   vocab_size: 98
   n_embd: 384
   n_head: 6
   n_layer: 6
   block_size: 256
   dropout: 0.35
   chars: 98 unique characters
   stoi: character to index mapping
   itos: index to character mapping

```


## Step 14: Inference - Loading and Using the Trained Model

Inference is the process of using a trained model to make predictions on new data.
Unlike training, inference:
- Does **not** compute gradients (faster and uses less memory)
- Uses the model in **evaluation mode** (disables dropout)
- Can process inputs of **variable length** (up to block_size)

### Why Save and Load?
1. **Avoid retraining**: Training takes time and compute resources
2. **Deployment**: Use the model in production applications
3. **Sharing**: Distribute trained models to others
4. **Checkpointing**: Save progress during long training runs

### The Inference Pipeline
1. Load the checkpoint file
2. Extract hyperparameters to reconstruct model architecture
3. Create a new model with the same architecture
4. Load the saved weights into the model
5. Set model to evaluation mode
6. Generate text from any starting prompt


```python
# determine device for inference
# this cell can be run independently after the model has been trained and saved
# note: torch, nn, and F are already imported from earlier cells
if torch.cuda.is_available():
    inference_device = 'cuda'
elif torch.backends.mps.is_available():
    inference_device = 'mps'
else:
    inference_device = 'cpu'

# display the selected inference device
print(f'inference device: {inference_device}')
```


**Output:**
```
inference device: mps

```


```python
# load the checkpoint
checkpoint_path = 'gpt_model.pt'
checkpoint = torch.load(checkpoint_path, map_location=inference_device, weights_only=False)

# extract hyperparameters from checkpoint
loaded_vocab_size = checkpoint['vocab_size']
loaded_n_embd = checkpoint['n_embd']
loaded_n_head = checkpoint['n_head']
loaded_n_layer = checkpoint['n_layer']
loaded_block_size = checkpoint['block_size']
loaded_dropout = checkpoint['dropout']

# extract tokenizer mappings
loaded_chars = checkpoint['chars']
loaded_stoi = checkpoint['stoi']
loaded_itos = checkpoint['itos']

# display checkpoint information
print('checkpoint loaded successfully')
print()
print('hyperparameters from checkpoint:')
print(f'   vocab_size: {loaded_vocab_size}')
print(f'   n_embd: {loaded_n_embd}')
print(f'   n_head: {loaded_n_head}')
print(f'   n_layer: {loaded_n_layer}')
print(f'   block_size: {loaded_block_size}')
print(f'   dropout: {loaded_dropout}')
```


**Output:**
```
checkpoint loaded successfully

hyperparameters from checkpoint:
   vocab_size: 98
   n_embd: 384
   n_head: 6
   n_layer: 6
   block_size: 256
   dropout: 0.35

```


```python
# define encode and decode functions using loaded tokenizer
def inference_encode(s):
    '''
    Convert a string to a list of integers using loaded tokenizer.
    
    Args:
        s: input string to encode
        
    Returns:
        list of integers representing each character
    '''
    # convert each character to its integer index
    return [loaded_stoi[c] for c in s]

def inference_decode(l):
    '''
    Convert a list of integers back to a string using loaded tokenizer.
    
    Args:
        l: list of integers to decode
        
    Returns:
        string representation of the integers
    '''
    # convert each integer back to its character and join
    return ''.join([loaded_itos[i] for i in l])

# display confirmation and test tokenizer functions
print('tokenizer functions created')
print()
print('testing tokenizer:')
test_str = 'hello'
encoded_test = inference_encode(test_str)
decoded_test = inference_decode(encoded_test)
print(f'   encode("{test_str}") = {encoded_test}')
print(f'   decode({encoded_test}) = "{decoded_test}"')
```


**Output:**
```
tokenizer functions created

testing tokenizer:
   encode("hello") = [63, 60, 67, 67, 70]
   decode([63, 60, 67, 67, 70]) = "hello"

```


```python
# InferenceHead: one head of self-attention for inference
class InferenceHead(nn.Module):
    '''
    Single head of self-attention for inference.
    
    Identical architecture to training Head class, but uses loaded hyperparameters
    from the checkpoint file instead of global variables.
    
    Computes scaled dot-product attention:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(head_size)) @ V
    
    Args:
        head_size: dimension of queries, keys, and values
    
    Attributes:
        key: linear projection for keys (what I contain)
        query: linear projection for queries (what I'm looking for)
        value: linear projection for values (what I'll give)
        tril: lower triangular mask for causal attention
        dropout: dropout layer for regularization
    '''
    
    def __init__(self, head_size):
        '''
        Initialize the attention head.
        
        Args:
            head_size: dimension of the attention head
        '''
        super().__init__()
        
        # key projection: (loaded_n_embd) -> (head_size)
        self.key = nn.Linear(loaded_n_embd, head_size, bias=False)
        
        # query projection: (loaded_n_embd) -> (head_size)
        self.query = nn.Linear(loaded_n_embd, head_size, bias=False)
        
        # value projection: (loaded_n_embd) -> (head_size)
        self.value = nn.Linear(loaded_n_embd, head_size, bias=False)
        
        # register_buffer: not a parameter, but should be saved with model
        self.register_buffer('tril', torch.tril(torch.ones(loaded_block_size, loaded_block_size)))
        
        # type hint for tril buffer
        self.tril: torch.Tensor
        
        # dropout for regularization
        self.dropout = nn.Dropout(loaded_dropout)
    
    def forward(self, x):
        '''
        Forward pass of the attention head.
        
        Args:
            x: input tensor of shape (B, T, C)
               B = batch size
               T = sequence length
               C = loaded_n_embd (embedding dimension)
        
        Returns:
            out: output tensor of shape (B, T, head_size)
        '''
        # get dimensions (only T is needed for masking)
        _, T, _ = x.shape
        
        # compute keys and queries
        k = self.key(x)
        q = self.query(x)
        
        # compute attention scores with scaling
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        
        # apply causal mask
        tril = self.tril
        wei = wei.masked_fill(tril[:T, :T] == 0, float('-inf'))
        
        # apply softmax and dropout
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # compute values and weighted aggregation
        v = self.value(x)
        out = wei @ v
        
        # return the attention output
        return out


# InferenceMultiHeadAttention: multiple heads of self-attention in parallel
class InferenceMultiHeadAttention(nn.Module):
    '''
    Multi-head self-attention for inference.
    
    Identical architecture to training MultiHeadAttention class, but uses loaded
    hyperparameters from the checkpoint file instead of global variables.
    
    Runs multiple attention heads in parallel and concatenates the results.
    Then projects back to the embedding dimension.
    
    Args:
        num_heads: number of attention heads
        head_size: dimension of each head
    
    Attributes:
        heads: ModuleList of InferenceHead modules
        proj: output projection back to loaded_n_embd
        dropout: dropout layer for regularization
    '''
    
    def __init__(self, num_heads, head_size):
        '''
        Initialize multi-head attention.
        
        Args:
            num_heads: number of parallel attention heads
            head_size: dimension of each attention head
        '''
        super().__init__()
        
        # create num_heads attention heads
        self.heads = nn.ModuleList([InferenceHead(head_size) for _ in range(num_heads)])
        
        # output projection: (num_heads * head_size) -> (loaded_n_embd)
        self.proj = nn.Linear(head_size * num_heads, loaded_n_embd)
        
        # dropout for regularization
        self.dropout = nn.Dropout(loaded_dropout)
    
    def forward(self, x):
        '''
        Forward pass of multi-head attention.
        
        Args:
            x: input tensor of shape (B, T, C)
        
        Returns:
            out: output tensor of shape (B, T, C)
        '''
        # run all heads in parallel and concatenate
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # project back to embedding dimension and apply dropout
        out = self.dropout(self.proj(out))
        
        # return the multi-head attention output
        return out


# InferenceFeedForward: simple feed-forward network (per token)
class InferenceFeedForward(nn.Module):
    '''
    Position-wise feed-forward network for inference.
    
    Identical architecture to training FeedForward class, but uses loaded
    hyperparameters from the checkpoint file instead of global variables.
    
    Applied to each position independently and identically.
    Expands to 4x the embedding dimension, applies GELU, then projects back.
    
    Args:
        n_embd: embedding dimension
    
    Attributes:
        net: Sequential network with linear, gelu, linear, dropout
    '''
    
    def __init__(self, n_embd):
        '''
        Initialize the feed-forward network.
        
        Args:
            n_embd: embedding dimension (input and output size)
        '''
        super().__init__()
        
        # build the feed-forward network
        self.net = nn.Sequential(
            # expand to 4x embedding dimension
            nn.Linear(n_embd, 4 * n_embd),
            # GELU non-linearity (smoother than ReLU, used in GPT-2/3)
            nn.GELU(),
            # project back to embedding dimension
            nn.Linear(4 * n_embd, n_embd),
            # dropout for regularization
            nn.Dropout(loaded_dropout),
        )
    
    def forward(self, x):
        '''
        Forward pass of the feed-forward network.
        
        Args:
            x: input tensor of shape (B, T, C)
        
        Returns:
            out: output tensor of shape (B, T, C)
        '''
        # apply the feed-forward network
        return self.net(x)


# InferenceBlock: transformer block with communication followed by computation
class InferenceBlock(nn.Module):
    '''
    Transformer block for inference: communication (attention) followed by computation (ffn).
    
    Identical architecture to training Block class, but uses loaded
    hyperparameters from the checkpoint file instead of global variables.
    
    Uses pre-norm architecture where layer norm is applied before each sub-layer.
    Residual connections allow gradients to flow directly through the network.
    
    Args:
        n_embd: embedding dimension
        n_head: number of attention heads
    
    Attributes:
        sa: self-attention module (multi-head attention)
        ffwd: feed-forward network
        ln1: first layer normalization
        ln2: second layer normalization
    '''
    
    def __init__(self, n_embd, n_head):
        '''
        Initialize the transformer block.
        
        Args:
            n_embd: embedding dimension
            n_head: number of attention heads
        '''
        super().__init__()
        
        # calculate head size: n_embd must be divisible by n_head
        head_size = n_embd // n_head
        
        # self-attention layer (communication)
        self.sa = InferenceMultiHeadAttention(n_head, head_size)
        
        # feed-forward layer (computation)
        self.ffwd = InferenceFeedForward(n_embd)
        
        # layer normalizations (pre-norm architecture)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        '''
        Forward pass of the transformer block.
        
        Args:
            x: input tensor of shape (B, T, C)
        
        Returns:
            out: output tensor of shape (B, T, C)
        '''
        # self-attention with residual connection
        x = x + self.sa(self.ln1(x))
        
        # feed-forward with residual connection
        x = x + self.ffwd(self.ln2(x))
        
        # return the transformed output
        return x


# InferenceGPT: the complete GPT model for inference
class InferenceGPT(nn.Module):
    '''
    GPT Language Model for inference.
    
    Identical architecture to training GPTLanguageModel class, but uses loaded
    hyperparameters from the checkpoint file instead of global variables.
    
    A decoder-only transformer that predicts the next token given previous tokens.
    
    Architecture:
        1. Token embedding: convert token indices to vectors
        2. Position embedding: add position information
        3. Transformer blocks: stack of attention + ffn blocks
        4. Final layer norm: normalize before output
        5. Language model head: project to vocabulary size for predictions
    
    Attributes:
        token_embedding_table: lookup table for token embeddings
        position_embedding_table: lookup table for position embeddings
        blocks: Sequential stack of transformer blocks
        ln_f: final layer normalization
        lm_head: linear projection to vocabulary size
    '''
    
    def __init__(self):
        '''
        Initialize the GPT language model for inference.
        '''
        super().__init__()
        
        # token embedding table: (loaded_vocab_size, loaded_n_embd)
        self.token_embedding_table = nn.Embedding(loaded_vocab_size, loaded_n_embd)
        
        # position embedding table: (loaded_block_size, loaded_n_embd)
        self.position_embedding_table = nn.Embedding(loaded_block_size, loaded_n_embd)
        
        # stack of transformer blocks
        self.blocks = nn.Sequential(*[InferenceBlock(loaded_n_embd, n_head=loaded_n_head) for _ in range(loaded_n_layer)])
        
        # final layer normalization
        self.ln_f = nn.LayerNorm(loaded_n_embd)
        
        # language model head: project from loaded_n_embd to loaded_vocab_size
        self.lm_head = nn.Linear(loaded_n_embd, loaded_vocab_size)
    
    def forward(self, idx, targets=None):
        '''
        Forward pass of the GPT model.
        
        Args:
            idx: input token indices of shape (B, T)
            targets: target token indices of shape (B, T), optional
        
        Returns:
            logits: unnormalized predictions of shape (B, T, loaded_vocab_size)
            loss: cross-entropy loss if targets provided, else None
        '''
        # get sequence length
        _, T = idx.shape
        
        # get token embeddings: (B, T) -> (B, T, loaded_n_embd)
        tok_emb = self.token_embedding_table(idx)
        
        # get position embeddings: (T,) -> (T, loaded_n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=inference_device))
        
        # combine token and position embeddings
        x = tok_emb + pos_emb
        
        # pass through transformer blocks
        x = self.blocks(x)
        
        # final layer norm
        x = self.ln_f(x)
        
        # project to vocabulary
        logits = self.lm_head(x)
        
        # compute loss if targets are provided
        if targets is None:
            loss = None
        else:
            # reshape for cross entropy
            _, _, C = logits.shape
            logits_flat = logits.view(-1, C)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        # return predictions and optional loss
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''
        Generate new tokens autoregressively.
        
        Given a context, generate max_new_tokens new tokens one at a time.
        
        Args:
            idx: starting context of shape (B, T)
            max_new_tokens: number of new tokens to generate
        
        Returns:
            idx: extended sequence of shape (B, T + max_new_tokens)
        '''
        # generate one token at a time
        for _ in range(max_new_tokens):
            # crop context to loaded_block_size
            idx_cond = idx[:, -loaded_block_size:]
            
            # get predictions
            logits, _ = self.forward(idx_cond)
            
            # focus on last time step
            logits = logits[:, -1, :]
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # append to running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        # return the extended sequence
        return idx

# display confirmation
print('inference model classes defined')
```


**Output:**
```
inference model classes defined

```


```python
# create inference model and load saved weights
inference_model = InferenceGPT()
inference_model.load_state_dict(checkpoint['model_state_dict'])
inference_model = inference_model.to(inference_device)

# set model to evaluation mode
# this disables dropout for deterministic inference
inference_model.eval()

# display model status and configuration
print('model loaded and ready for inference')
print()
print(f'model device: {inference_device}')
print(f'model mode: evaluation (dropout disabled)')
print(f'max context length: {loaded_block_size} tokens')
```


**Output:**
```
model loaded and ready for inference

model device: mps
model mode: evaluation (dropout disabled)
max context length: 256 tokens

```


### Generate Text w/ Variable Context Length

You can provide any starting prompt (context) and the model will continue generating from there. The context can be any length from 1 character up to `block_size` characters.


```python
# generate text with variable context length
# change these variables to customize generation

# starting prompt (context) - can be any text from the training data's character set
# the model will continue generating from this starting point
prompt = 'The '

# number of new tokens to generate
num_tokens_to_generate = 500

# validate prompt contains only known characters
for char in prompt:
    if char not in loaded_stoi:
        print(f'warning: character "{char}" not in vocabulary, replacing with space')
        prompt = prompt.replace(char, ' ')

# encode the prompt
context_tokens = inference_encode(prompt)
context_length = len(context_tokens)

# display prompt information
print(f'prompt: "{prompt}"')
print(f'context length: {context_length} tokens')
print(f'tokens to generate: {num_tokens_to_generate}')
print()

# convert to tensor and move to device
context_tensor = torch.tensor([context_tokens], dtype=torch.long, device=inference_device)

# generate with no gradient computation (faster and uses less memory)
with torch.no_grad():
    generated_tokens = inference_model.generate(context_tensor, max_new_tokens=num_tokens_to_generate)

# decode and display
generated_text = inference_decode(generated_tokens[0].tolist())

# display the generated text
print('generated text:')
print('=' * 60)
print(generated_text)
print('=' * 60)
```


**Output:**
```
prompt: "The "
context length: 4 tokens
tokens to generate: 500

generated text:
============================================================
The panies he
has easy, and Mr. James, we have done me on the morning. If you will be
break if you from anything that I may still of making my absolutely commonplace easy
at Briony Lodge, this is quite as to the furniture of the room above, this
vacancy, which you will not remain the meantime of whom we wish to a
requestion to open the struggle. There is one that the day which you are
sure this morning end.”

“And you will know how question. I did not best to be asticle enough.”

“You know, I fancy,
============================================================

```


```python
# reusable function for generating text with any prompt
def generate_text(prompt='', num_tokens=200):
    '''
    Generate text from the trained model.
    
    Args:
        prompt: starting text (empty string starts from scratch)
        num_tokens: number of new tokens to generate
    
    Returns:
        generated text string
    '''
    # handle empty prompt
    if prompt == '':
        context_tensor = torch.zeros((1, 1), dtype=torch.long, device=inference_device)
    else:
        # validate and encode prompt
        valid_prompt = ''
        for char in prompt:
            if char in loaded_stoi:
                valid_prompt += char
            else:
                valid_prompt += ' '
        context_tokens = inference_encode(valid_prompt)
        context_tensor = torch.tensor([context_tokens], dtype=torch.long, device=inference_device)
    
    # generate with no gradients
    with torch.no_grad():
        generated = inference_model.generate(context_tensor, max_new_tokens=num_tokens)
    
    # decode and return the generated text
    return inference_decode(generated[0].tolist())

# display function usage and example
print('generate_text function defined')
print()
print('usage:')
print('   generate_text(prompt="The ", num_tokens=200)')
print('   generate_text(prompt="", num_tokens=500)  # start from scratch')
print()
print('example:')
print('-' * 40)
sample = generate_text(prompt='What is your name?', num_tokens=50)
print(sample)
print('-' * 40)
```


**Output:**
```
generate_text function defined

usage:
   generate_text(prompt="The ", num_tokens=200)
   generate_text(prompt="", num_tokens=500)  # start from scratch

example:
----------------------------------------
What is your name?”

“I should be be good for passional jest for thi
----------------------------------------

```


## Summary: What We Built

We implemented a complete GPT language model from scratch with the following components:

| Component | Purpose |
|-----------|---------|
| **Token Embedding** | Convert token indices to dense vectors |
| **Position Embedding** | Add position information to tokens |
| **Multi-Head Attention** | Tokens communicate with each other |
| **Feed-Forward Network** | Process each token independently (with GELU activation) |
| **Layer Normalization** | Stabilize training (pre-norm architecture) |
| **Residual Connections** | Enable deep networks |
| **Model Saving** | Save trained weights to disk |
| **Inference Pipeline** | Load and use model for generation |

### The Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Our Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_embd` | 384 | Embedding dimension |
| `n_head` | 6 | Number of attention heads |
| `n_layer` | 6 | Number of transformer blocks |
| `block_size` | 256 | Maximum context length |
| `batch_size` | 64 | Training batch size |
| `dropout` | 0.35 | Regularization |
| `patience` | 5 | Early stopping patience (evaluations) |
| `max_iters` | 10,000 | Maximum training iterations |
| `learning_rate` | 5e-4 | Initial learning rate |

### Training Techniques

| Technique | Implementation | Purpose |
|-----------|----------------|---------|
| **AdamW Optimizer** | `weight_decay=0.15` | L2 regularization to prevent overfitting |
| **Cosine LR Scheduler** | 5e-4 → 1e-5 | Smooth learning rate decay for better convergence |
| **Gradient Clipping** | `max_norm=1.0` | Prevent exploding gradients |
| **GELU Activation** | `nn.GELU()` | Smoother gradients than ReLU (used in GPT-2/3) |
| **Early Stopping** | `patience=5` | Stop training when val loss stops improving |

### GPU Acceleration Support

| Platform | Device | How It's Used |
|----------|--------|---------------|
| Windows/Linux | NVIDIA CUDA | `torch.cuda.is_available()` |
| Mac (M1/M2/M3) | Apple MPS | `torch.backends.mps.is_available()` |
| Any | CPU | Fallback when no GPU available |

### Model Persistence

The trained model is saved with:
- **model_state_dict**: All learned weights and biases
- **Hyperparameters**: Architecture configuration
- **Tokenizer**: Character mappings for encode/decode

Congratulations! You have built a complete GPT from scratch with training, saving, and inference!


## MIT License


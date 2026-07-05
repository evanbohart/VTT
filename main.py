from model.model import Model
from data.data import Data
from train import train

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader

batch_size = 4
accum_steps = 8

n_fft = 400
hop_len = 160
n_mels = 80
encoder_seq_len = 1000
decoder_seq_len = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = Data(
    torchaudio.datasets.LIBRISPEECH(
        root='.',
        url='train-clean-100',
        download=True
    ),
    n_fft,
    hop_len,
    n_mels,
    encoder_seq_len,
    decoder_seq_len,
)

loader = DataLoader(
data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data.collate_fn,
)

d_model = 512
h = 8
d_k = d_v = d_model // h
d_ff = 2048
n = 6
dropout = 0.1

vocab_size = len(data.vocab)

model = Model(
    encoder_x_dim=n_mels,
    vocab_size=vocab_size,
    d_model=d_model,
    encoder_seq_len=encoder_seq_len,
    decoder_seq_len=decoder_seq_len,
    h=h,
    d_k=d_k,
    d_v=d_v,
    d_ff=d_ff,
    n=n,
    dropout=dropout
)

model = model.to(device)
model.train()
nn.utils.clip_grad_norm(model.parameters(), 1.0)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9,0.98), eps=1e-9, weight_decay=0.01)

def lr_lambda(step):
    warmup_steps = 1000

    if step > warmup_steps:
        return 1.0

    return step / warmup_steps

scheduler = optim.lr_scheduler.LRScheduler(optimizer, lr_lambda=lr_lambda)

scaler = torch.amp.GradScaler("cuda")

epochs = 100

train(
    epochs,
    accum_steps,
    loader,
    model,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device
)

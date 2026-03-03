import torch
import numpy as np
from gpt import GPTModel
import matplotlib.pyplot as plt


# since we didn't really cover how to do this in lecture-
# this creates a learning rate schedule for you. Refer to the
# pytorch docs for more info on using a scheduler.

# This one is designed for you to call scheduler.step() on every
# model update step.
def cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps):
    def thunk(stepnum):
        if stepnum <= warmup_steps:
            # go from ~0 to 1.0
            prog = float(stepnum)/float(warmup_steps)
            lrmult = 0.00001 + prog
        else:
            # go from 1.0 to ~0
            steps_after_peak = stepnum-warmup_steps
            tail_steps = total_steps-warmup_steps
            prog = float(steps_after_peak) / float(tail_steps)
            lrmult = ((np.cos(3.141592*prog)+1.0)*0.5)*0.9 + 0.1
        return max(lrmult, 0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=thunk)
    return scheduler

# ===========================================================================

'''
Training hyperparameters selected:
  - Model:       d_model=512, n_heads=8, layers=6, vocab_size=10000, max_seq_len=256
  - Batch size:  32
  - Peak LR:     3e-4  (AdamW)
  - Warmup:      500 steps
  - Grad clip:   1.0
  - Sequence length: 256 (set in construct_dataset.py)
  - Loss logged every 100 steps as loss_curve.png
'''
def train():

    device = torch.device("cuda") # use "cpu" if no gpu available

    # --- hyperparameters ---
    batch_size   = 32
    learning_rate = 3e-4
    warmup_steps  = 500
    log_interval  = 100   # save loss curve every N steps

    model = GPTModel(d_model=512, n_heads=8, layers=6, vocab_size=10000, max_seq_len=256)
    param_count = sum(p.numel() for p in model.parameters())
    print("Model has", param_count, "parameters.")

    model = model.to(device)

    # load packed dataset produced by construct_dataset.py
    data = np.load("./dataset.npy")
    data = torch.tensor(data, dtype=torch.long)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=batch_size,
        shuffle=True,
    )

    total_steps = len(dataloader)
    print(f"Total steps per epoch: {total_steps}")

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    tokens_seen = 0
    tokens_log  = []
    loss_log    = []

    for step, (batch,) in enumerate(dataloader):
        batch = batch.to(device)
        x = batch[:, :-1]   # (B, S)   — input tokens
        y = batch[:, 1:]    # (B, S)   — target tokens (shifted by 1)

        opt.zero_grad()

        logits = model(x)   # (B, S, vocab_size)

        # CrossEntropyLoss expects class scores in dim 1: (B, vocab_size, S)
        loss = loss_fn(logits.transpose(1, 2), y)

        loss.backward()

        # clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step the optimizer and scheduler
        opt.step()
        scheduler.step()

        # log total tokens and loss
        B, S = x.shape
        tokens_seen += B * S

        if step % log_interval == 0:
            tokens_log.append(tokens_seen)
            loss_log.append(loss.item())
            print(f"Step {step:>6} | Tokens {tokens_seen:>12,} | Loss {loss.item():.4f}")

            # periodically save a plot of loss vs tokens
            plt.figure()
            plt.plot(tokens_log, loss_log)
            plt.xlabel("Tokens Processed")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("GPT Training Loss")
            plt.tight_layout()
            plt.savefig("./loss_curve.png")
            plt.close()

    # save model weights if you want
    torch.save(model.state_dict(), "./model_weights.pt")
    print("Training complete. Weights saved to model_weights.pt")



if __name__ == "__main__":
    train()

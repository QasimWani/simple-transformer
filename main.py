import math

import torch
import torch.nn as nn
from datasets import load_dataset
from model import SimpleGPT, device
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer


class WikiTextDataset(Dataset):
    def __init__(self, split="train", max_length=128, subset_size=None):
        """
        Args:
            split (str): "train", "validation", or "test"
            max_length (int): maximum sequence length
            subset_size (int): optional, number of examples to sample
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Wikipedia dataset (March 2022 English dump)
        dataset = load_dataset("wikipedia", "20220301.en", split=split)

        # Optionally shrink dataset for quicker experiments
        if subset_size:
            dataset = dataset.shuffle(seed=42).select(range(subset_size))

        # Pre-tokenize all texts
        def tokenize(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        dataset = dataset.map(tokenize, batched=True, remove_columns=["id", "url", "title", "text"])
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": item["input_ids"],         # (max_length,)
            "attention_mask": item["attention_mask"]  # (max_length,)
        }


def get_wiki_dataloader(split="train", batch_size=8, max_length=128, subset_size=None):
    dataset = WikiTextDataset(split=split, max_length=max_length, subset_size=subset_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



def compute_loss(logits, input_ids, attention_mask):
    """
    Shift for next-token prediction and ignore pads in the loss.
    logits: (batch_size, seq_len, vocab_size)
    input_ids: (batch_size, seq_len)
    attention_mask: (batch_size, seq_len) with 1=token, 0=pad
    """
    # shift: predict token t+1 from positions up to T-1
    logits = logits[:, :-1, :]  # (batch_size, seq_len - 1, vocab_size)
    labels = input_ids[:, 1:].clone()  # (batch_size, seq_len - 1)
    keep = attention_mask[:, 1:]  # (batch_size, seq_len - 1)

    # set <pad> token to ignore_index so we skip it during loss calculation
    labels[keep == 0] = -100
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    return loss


def train(dataloader: DataLoader, max_steps: int = 2000, lr: float = 3e-4):
    model = SimpleGPT().to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Optional: cosine decay with linear warmup (last 10% decay)
    warmup_steps = max(10, int(0.05 * max_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    step = 0
    data_iter = iter(dataloader)
    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        optimizer.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(device)  # (batch_size, seq_len)
        attention_mask = batch["attention_mask"].to(device)  # (batch_size, seq_len)

        logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
        loss = compute_loss(logits, input_ids, attention_mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1
        if step % 50 == 0:
            print(f"step {step}/{max_steps}  loss {loss.item():.4f}  lr {scheduler.get_last_lr()[0]:.2e}")

    return model

@torch.no_grad()
def generate(model: SimpleGPT, tokenizer, prompt: str, max_new_tokens=64, temperature=1.0, top_k=None):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    for _ in range(max_new_tokens):
        logits = model(input_ids)[:, -1, :] / max(1e-8, temperature)
        if top_k is not None:
            # keep top_k logits, set rest to -inf
            v, _ = torch.topk(logits, k=top_k)
            thresh = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < thresh, torch.full_like(logits, -float("inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if input_ids.size(1) >= model.max_seq_len:
            break
    return tokenizer.decode(input_ids[0].tolist())


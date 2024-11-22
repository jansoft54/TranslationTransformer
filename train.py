import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from model import TranslationModel
from lr_scheduler import InverseSquareRootLR

device = "cuda"
dataset = load_dataset("wmt14", "de-en")

dataset['train'] = dataset['train'].select(range(10000))
dataset['test'] = dataset['test'].select(range(1000))
dataset['validation'] = dataset['validation'].select(range(1000))

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
src_seq_len = 128
trg_seq_len = 128
batch_size=256

def preprocess_function(examples):
    inputs = [item["de"] for item in examples["translation"]]
    targets = [item["en"] for item in examples["translation"]]
    labels = [item["en"] for item in examples["translation"]]

    model_inputs = tokenizer(inputs, padding="max_length", truncation=True,max_length=src_seq_len,add_special_tokens=False)
    model_inputs['src_mask'] = model_inputs["attention_mask"]

    with tokenizer.as_target_tokenizer():
        targets = tokenizer(targets, padding="max_length", truncation=True,max_length=trg_seq_len,add_special_tokens=True)
        labels = tokenizer(labels, padding="max_length", truncation=True,max_length=trg_seq_len,add_special_tokens=True)


    model_inputs["target"] = targets["input_ids"]
    model_inputs["target_mask"] = targets["attention_mask"]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_data = dataset.map(preprocess_function, batched=True)
tokenized_data.set_format(type='torch', columns=['input_ids', 'src_mask', 'target','target_mask','labels'])

train_dataloader = DataLoader(
    tokenized_data["train"],
    batch_size=batch_size,
    shuffle=True,
)


epochs = 100
warmup = 10
lr = 2.5e-4
min_lr = 1e-6


model = TranslationModel(vocab_size=tokenizer.vocab_size,d_model=512,src_seq_len=src_seq_len,trg_seq_len=trg_seq_len)
model.to(device)
optim = torch.optim.AdamW(model.parameters(),lr=lr,betas=(0.9, 0.98))
lr_scheduler = InverseSquareRootLR(optim,warmup,lr,min_lr=min_lr)

loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
for epoch in range(epochs):
    for batch in train_dataloader:
        src = batch["input_ids"].to(device)
        src_attention_mask = batch["src_mask"].to(device)
        tgt = batch["target"].to(device)
        tgt_attention_mask = batch["target_mask"].to(device)

        labels = batch["labels"].to(device)[:,1:]

        pad_tokens = torch.full((labels.shape[0], 1), tokenizer.pad_token_id, dtype=tgt.dtype).to(device)

        labels = torch.cat((labels, pad_tokens), dim=-1)
        output = model(src,tgt,src_attention_mask,tgt_attention_mask)


        output = output.view(-1,output.shape[-1])
        labels = labels.view(-1)

        loss = loss_fn(output,labels)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optim.step()
        lr_scheduler.step()
    print(loss)

from embedding import *;
from model import *
import json;
import torch
from torch import nn;
import os
import data as dt;
import matplotlib.pyplot as plt;
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = torch.device("mps")

epochs = 50;
batch_size=32;
mode = 1

with open("dataset.json","r") as f:
    data = json.load(f);
with open("test.json","r") as f:
    test = json.load(f);
with open("val.json","r") as f:
    val = json.load(f);
src = [i["input"] for i in data];
tgt = [i["output"][::-1] for i in data];
src_test = [i["input"] for i in test];
tgt_test = [i["output"][::-1] for i in test];
src_val = [i["input"] for i in val];
tgt_val = [i["output"][::-1] for i in val];


embedding = Embedding();
if(mode==0):
    model = Model(embedding).to(device);
if(mode==1):
    model = TransformerByHand(embedding).to(device)
if(mode==2):
    model = DecoderOnly(embedding).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
def evaluate(src_test,tgt_test):
    model.to("cpu")
    model.eval()
    with torch.no_grad():
        #print(src_test[0],end='')
        src_test_tokens = embedding.tokenize(src_test,pad=(mode!=2)).to("cpu");
        sos_id = embedding.words["<SOS>"]
        eos_id = embedding.words["<EOS>"]

        tgt_indices = torch.LongTensor([[sos_id]]).to("cpu")
        max_len = 30  # 防止死循环，设定最大生成长度
        result_ids = []
        for i in range(max_len):
            if(mode==2):
                tokens_in = torch.cat([src_test_tokens,tgt_indices],dim=1);
                logits = model(tokens_in);
            else:
                logits = model(src_test_tokens,tgt_indices);
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            if next_token == eos_id:
                break

            result_ids.append(next_token)
            new_token_tensor = torch.LongTensor([[next_token]]).to("cpu")
            tgt_indices = torch.cat([tgt_indices, new_token_tensor], dim=1)

        id_to_word = {v: k for k, v in embedding.words.items()}
        prediction = "".join([id_to_word[idx] for idx in result_ids])
        model.to(device)
        #print(prediction[::-1]);
        return prediction == tgt_test;
acc=[];
for epoch in range(epochs):
    model.train()
    for i in range(0,len(src),batch_size):
        if(mode==2):
            batch_src = [list(j) for j in src[i:i + batch_size]];
            batch_tgt = [["<SOS>"] + list(j) + ["<EOS>"] for j in tgt[i:i + batch_size]];
            batch_words = []
            for j in range(len(batch_src)):
                batch_words.append(batch_src[j] + batch_tgt[j]);
            batch_tokens = embedding.tokenize(batch_words).to(device);
            tokens_in = batch_tokens[:,:-1];
            tgt_out = batch_tokens[:,len(batch_src[0])+1:]
            output = model(tokens_in)[:, len(batch_src[0]):, :]
            loss = criterion(output.reshape(-1,len(embedding.words)),tgt_out.reshape(-1))
        else:
            src_tokens = embedding.tokenize(src[i:i + batch_size]).to(device)
            tgt_tokens = embedding.tokenize(tgt[i:i + batch_size], is_target=True).to(device);
            tgt_in = tgt_tokens[:, :-1]
            tgt_out = tgt_tokens[:, 1:]
            output = model(src_tokens,tgt_in);
            loss = criterion(output.reshape(-1, len(embedding.words)), tgt_out.reshape(-1))
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    if (epoch+1) % 5 == 0:
        n = len(src_val);
        c = 0;
        for i, t in enumerate(src_val):
            c += evaluate([t], tgt_val[i]);
        print(f"Epoch {epoch+1}, Loss: {loss.item()}, val_accuracy: {c}/{n}")
        acc.append(c/n);

# plt.plot(list(range(1,epochs+1,5)), acc, marker='o', linestyle='-', color='b')
# plt.xlabel('epoch')       # X轴说明
# plt.ylabel('accuracy')       # Y轴说明
#
# plt.title('epoch/accuracy(num_train=%s)'%dt.num_train)
#
# plt.savefig("overview.png")

n = len(src_test);
c = 0;
for i,t in enumerate(src_test):
    c += evaluate([t],tgt_test[i]);

print(f"{c}/{n}")

from torch import nn;
import torch;
import math;
class Embedding(nn.Module):
    def __init__(self,dim=128, max_len=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim;
        self.words = {j:(i+3) for i,j in enumerate("0123456789+=")}
        self.words["<PAD>"]=0;self.words["<SOS>"]=1;self.words["<EOS>"]=2;
        self.max_len = max_len
        self.token_embedding = nn.Embedding(len(self.words), self.dim);
        self.position_embedding = nn.Embedding(self.max_len,self.dim);

    def tokenize(self,batch_data,is_target=False,pad=True):
        sentences = []
        self.max_len=0
        for i in batch_data:
            self.max_len = max(self.max_len,len(i));
        self.max_len+=is_target*2
        for sentence in batch_data:
            tokens = [self.words[word] for word in sentence]
            if(is_target):
                tokens = [self.words["<SOS>"]]+tokens+[self.words["<EOS>"]]
            if(pad):
                tokens+=[self.words["<PAD>"]]*(self.max_len-len(tokens));
            sentences.append(tokens)
        return torch.LongTensor(sentences);

    def forward(self,x):
        # x:batch_size seq_len
        batch_size,seq_len = x.size();
        positions = torch.arange(0,seq_len).expand(batch_size,seq_len).to(x.device) #用torch在一个batch里做序列位置编码
        out = self.token_embedding(x)* math.sqrt(self.dim)+self.position_embedding(positions)
        return out;




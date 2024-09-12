import torch
from torch import nn

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGramModel, self).__init__()
        self.embed_v = nn.Embedding(vocab_size, embed_size)
        self.embed_u = nn.Embedding(vocab_size, embed_size)

    def forward(self, center, contexts_and_negatives):
        return skip_gram(center, contexts_and_negatives, self.embed_v, self.embed_u)

if __name__ == "__main__":
    # 测试代码
    vocab_size = 10000
    embed_size = 100
    model = SkipGramModel(vocab_size, embed_size)
    
    # 创建一些假数据进行测试
    center = torch.randint(0, vocab_size, (64, 1))
    contexts_and_negatives = torch.randint(0, vocab_size, (64, 20))
    
    output = model(center, contexts_and_negatives)
    print("Output shape:", output.shape)
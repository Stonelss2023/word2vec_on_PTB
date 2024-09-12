import torch
from torch import nn
import time
from DL10_3_2_preprocess import get_preprocessed_data
from DL10_3_3_dataset_sampling import load_ptb_data
from DL10_3_4_model import SkipGramModel

class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
    
    def forward(self, inputs, targets, mask=None):
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        return res.mean(dim=1)

def train(net, data_iter, loss, num_epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [d.to(device) for d in batch]
            pred = net(center, context_negative)
            l = (loss(pred.view(label.shape), label, mask) *
                 mask.shape[1] / mask.float().sum(dim=1)).mean()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1
        print(f'epoch {epoch + 1}, loss {l_sum / n:.4f}, time {time.time() - start:.2f}s')
    
    return net

if __name__ == "__main__":
    # 数据预处理
    centers, contexts, negatives, vocab_info = get_preprocessed_data()
    
    # 创建数据迭代器
    batch_size = 128
    num_workers = 0  # 根据需要调整
    data_iter = load_ptb_data(centers, contexts, negatives, batch_size, num_workers)
    
    # 初始化模型
    embed_size = 50
    net = SkipGramModel(len(vocab_info['idx_to_token']), embed_size)
    
    # 定义损失函数
    loss = SigmoidBinaryCrossEntropyLoss()
    
    # 训练模型
    num_epochs = 50
    lr = 0.005
    trained_net = train(net, data_iter, loss, num_epochs, lr)

    # 这里可以添加模型评估、保存等其他逻辑
    
    print("Training completed.")
        # 添加这些新行来展示词相似度
    from DL10_3_5_similarity import get_similar_tokens

    # 测试几个词的相似度
    test_words = ['chip', 'computer', 'technology']
    for word in test_words:
        print(f"\nSimilar words to '{word}':")
        get_similar_tokens(word, 3, trained_net.embed_v, 
                           vocab_info['idx_to_token'], 
                           vocab_info['token_to_idx'])
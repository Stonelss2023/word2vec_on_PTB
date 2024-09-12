import torch

def get_similar_tokens(query_token, k, embed, idx_to_token, token_to_idx):
    W = embed.weight.data
    x = W[token_to_idx[query_token]]
    # 添加1e-9是为了数值稳定性
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
    _, topk = torch.topk(cos, k=k+1)
    topk = topk.cpu().numpy()
    for i in topk[1:]:  # 除去输入词
        print(f'cosine sim={cos[i]:.3f}: {idx_to_token[i]}')


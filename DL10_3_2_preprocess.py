import collections
import random
import math
from DL10_3_1_download_word_embedding import load_ptb_data

def get_preprocessed_data(max_window_size=10, min_count=3, num_noise_words=8):
    # 加载数据
    dataset = load_ptb_data()
    
    # 构建词表
    counter = collections.Counter([tk for st in dataset for tk in st])
    counter = dict(filter(lambda x: x[1] >= min_count, counter.items())) 

    idx_to_token = [tk for tk, _ in counter.items()]
    token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
    dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
               for st in dataset]
    num_tokens = sum(len(st) for st in dataset)

    # 二次采样
    def discard(idx):
        return random.uniform(0, 1) < 1 - math.sqrt(
            1e-4 / counter[idx_to_token[idx]] * num_tokens)

    subsampled_dataset = [[idx for idx in st if not discard(idx)] for st in dataset]

    # 提取中心词和上下文词
    def get_centers_and_contexts(dataset, max_window_size):
        centers, contexts = [], []
        for st in dataset:
            if len(st) < 2:
                continue
            centers += st
            for center_i in range(len(st)):
                window_size = random.randint(1, max_window_size)
                indices = list(range(max(0, center_i - window_size),
                                     min(len(st), center_i + 1 + window_size)))
                indices.remove(center_i) 
                contexts.append([st[idx] for idx in indices])
        return centers, contexts

    all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, max_window_size)

    # 负采样
    def get_negatives(all_contexts, sampling_weights, K):
        all_negatives, neg_candidates, i = [], [], 0
        population = list(range(len(sampling_weights)))
        for contexts in all_contexts:
            negatives = []
            while len(negatives) < len(contexts) * K:
                if i == len(neg_candidates):
                    i, neg_candidates = 0, random.choices(
                        population, sampling_weights, k=int(1e5))
                neg, i = neg_candidates[i], i + 1
                if neg not in set(contexts):
                    negatives.append(neg)
            all_negatives.append(negatives)
        return all_negatives

    sampling_weights = [counter[w]**0.75 for w in idx_to_token]
    all_negatives = get_negatives(all_contexts, sampling_weights, num_noise_words)

    return all_centers, all_contexts, all_negatives, {
        'idx_to_token': idx_to_token,
        'token_to_idx': token_to_idx,
        'counter': counter,
        'subsampled_dataset': subsampled_dataset
    }

if __name__ == "__main__":
    # 测试代码
    centers, contexts, negatives, vocab_info = get_preprocessed_data()
    
    print(f"Number of center words: {len(centers)}")
    print(f"Number of context sets: {len(contexts)}")
    print(f"Number of negative sample sets: {len(negatives)}")
    print(f"Vocabulary size: {len(vocab_info['idx_to_token'])}")
    
    # 示例：打印前5个中心词及其上下文
    for i in range(5):
        print(f"Center word: {vocab_info['idx_to_token'][centers[i]]}")
        print(f"Context words: {[vocab_info['idx_to_token'][idx] for idx in contexts[i]]}")
        print(f"Negative samples: {[vocab_info['idx_to_token'][idx] for idx in negatives[i]]}")
        print()

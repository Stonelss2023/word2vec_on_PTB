PTB(Penn Tree Bank) -- 采样自《华尔街日报》文章的小型语料库

- 词级别数据集的preprocessing
    -词表构建
    -二次采样
    -提取中心词和背景词
    - 负采样

preprocessing期望得到的输出：centers, contexts, negatives三个元素相对应的集合,一些中间产物可以用打包在一个字典里
- 采样、批处理与数据封装
    -封装比较散,主要依赖Pytorch提供的Dataset类加载(自定义继承类)+DataLoader创建标准迭代器
    -批次中样本长度不同需要padding归一tensor形状,同时生成masks和labels便于训练过程提效
    -单独定义的批处理逻辑传给DataLoader的collate_fn参数
    -通过DataLoader的num_workers参数定义并行进程数(数据批次加载数)

- 读取数据
- 跳字模型
    -嵌入层
    -小批量乘法
    - 跳字模型前向计算
- 模型训练
    -二元交叉熵损失函数
    -初始化模型参数
    -定义训练函数
- 应用词嵌入模型
    
# 0. Download the dataset


```python
import torch
import torch.utils.data as Data
import sys
import os
import nltk

sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(torch.cuda.is_available())

def download_ptb():
    data_dir = "../../data/ptb"
    file_path = os.path.join(data_dir, 'ptb.train_txt')

    if os.path.exists(file_path):
        print("PTB dataset already exists. Loading from file...")
        with open(file_path, 'r', encoding='uft-8') as f:
            lines = f.readlines()
            raw_dataset = [st.split() for st in lines]
        print('# sentences:', len(raw_dataset))
        return raw_dataset

    #如果文件不存在→下载并处理数据
    #创建目录
    os.makedirs(data_dir, exist_ok=True)
    print("Downlaoding Treebank dataset...")
    nltk.dawnload('treebank')

    from nltk.corpus import treebank

    print(("Saving training data...")
    with oprn(file_path, 'w', encoding='utf-8') as f:
        for sent in treebank.sents():
            f.write(' '.jion(sent + '\n')
    print("Download and extraction complete.")
    print("PTB dataset is ready in", data_dir)

    # 读取训练数据                
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        raw_dataset = [st.split() for st in lines]

    print('# sentences:', len(raw_dataset))

    return rwa_dataset

def load_ptb_data():
    dataset = download_ptb()
    print(f"Dataset loaded with {len(dataset)}sentences."
    return dataset
                
```

# 1. Preprocessing
贯穿整个prjoect的线索是: Token Frequency!


```python
import collections
import math

def get_preprocessed_data(max_window_size=8, min_count=4, num_noise_words=24):
    dataset = load_ptb_data()

    # Establish the lexicon
    counter = collections.Counter([tk for st in dataset for tk in st])
    counter = dict(filter(lambda x: x[1] >= min_count, counter.items()))

    idx_to_token = [tk for tk, _ in counter.items()]
    token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
    dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx] 存疑
               for st in dataset]
    num_tokens = sum([len(st) for st in dataset]) 存疑

    # Second Sampling
    def discard(idx):
        word_frequency = counter[idx_to_token[idx]] / num_tokens
        return random.uniform(0, 1) < 1 - math.sqrt(
            5e-5 / word_frequency)

    subsampled_dataset = [[idx for idx in st if not discard(idx)] for st in dataset] 

    # Extracting centers and contexts
    def get_centers_and_contexts(dataset, max_window_size):
        centers, contexts = [], []
        for st in dataset:
            if len(st) < 2:
                continue
            centers += st
            for center_i in range(len(st)):
                window_size = random.randint(2, max_window_size)
                indices = list(range(max(0, center_i - window_size),
                                      min(len(st), center_i + 1 + window_size)))
                indices.remove(center_i)
                contexts.append([st[idx] for idx in indices])
        return centers, contexts

    all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, max_window_size)


    # Negative Sampling
    def get_negatives(all_contexts, sampling_weights, K):
        all_neagtives, neg_condidates, i = [], [], 0
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
    all_neagtives = get_neagtives(all_contexts, sampling_weights, num_noise_words)

    return all_centers, all_contexs, all_negatives, {
        'idx_to_token': idx_to_token,
        'token_to_idx': token_to_idx,
        'counter': counter,
        'subsampled_dataset': subasampled_dataset
    }

    
                
```

## 1) Lexicon Establishment
1. Counter对象是字典的一个子类，一个专门用于计数的字典类型
2. lambda函数接受一个参数x(此处为键值对），并检查x[1](值）是否 >=4
3. counter.item()返回一个由键值对组成的列表 .item() does not preserve original order
4. filter函数使用lambda过滤counter.item(), dict将过滤后的结果转回字典

词典只是一个查找工具，实际用于模型训练的Dataset保留了原始序列信息        

.items()方法是字典的标准方法返回键值对视图; .item()方法是pytorch张量方法用于获取只包含单一元素的张量的值,返回一个python标量

## 2) Second Sampling
1. random.uniform(0, 1) → 生成0-1之间的均匀分布；理论上说, 词频越高, 被丢弃的可能性越大
2. return 的是丢弃部分; subsampled_dataset 数据结构仍然是indice list

## 3) Extracting centers and contexts
1. random.randint → 生成离散均匀分布的随机整数(discrete uniform distribution)
2. max(0, center_i - window_size) 确保左边界不会小于0 ; min(len(st), center_i) 

## 4) Negative_Sampling
1. 从all_negatives列表逻辑中单独拉出来一个negatives列表逻辑，旨在为每段contexts窗口的负采样创造独立性
2. 最后return 返回分两部分
-  all_centers, all_contexs, all_negtives这三个是主要的处理结果
-  词索引列表和词典属于中间生成的辅助数据结构,放在一个字典里方便访问必要信息

# 2. Dataset Sampling


```python
import torch.utils.data as Data

class PTBDataset(Data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives) #数据结构不一样 → contexts&negatives的元素是lists
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], slef.contexts[index], dself.negatives[index])

    def __len__(self):
        return len(self.centers)

def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data) #计算所有样本中耽搁上下文和负样本tokens的最大值
    centers, contexts_negtives, masks, labels = [], [], [], []
    for center, context, negative in data: #data数据是__getitem__返回的元组包(c, c, n)
        cur_len = len(context) + len(negative)
        centers.append(center)
        contexts_negatives.append(context + negative + [0] * (max_len - cur_len))
        masks.append([1] * cur_len + [0] * (max_len - cur_len)) #掩码补全到同一长度
        labels.append([1] * len(context) + [0] * (max_len - len(contexts))) #标记正样本
    return (torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives),
            torch.tensor(masks), torch.tensor(labels))

def load_ptb_data(all_centers, all_contes, all_negatives, batch_size, num_workers):
    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True,
                                collate_fn=batchify,
                                num_workers=num_workers)

    return data_iter
  
```

1. assert 断言检查，检查条件是否为真。若返回false会引发assertionError异常
助力数据集在初始化阶段就发现潜在的数据问题，快速定位异常发生位置
assert断言检查主要用于调试目的，一行代码清晰地表达代码的预期条件，替代多行条件检查和错误处理代码

2. __getitem__()函数定义如何获取数据集中单个项 → 返回指定索引的中心词、上下文、负样本
在python中，以__开头结尾的方法被称为“魔术方法”→__getitem__方法定义了对象如何响应索引操作。其实，平常使用方括号语法访问对象时(如object[index])，Python会自动调用getitem方法，此方法是python中实现可索引对象的标准方式

此外，在Pytorch中，DataLoader依赖于这个方法访问数据集中的个别样本。没有这个方法就无法使用诸如dataset[i]这样的语法，也无法与DataLoader兼容

【补充解释与DataLoader的隐式交互】当遍历data_iter的时候，DataLoader会在后台进行以下操作：
(1) 基于batch_size生成一批索引
(2) 对于每个索引，调用dataset.getitem(index)
(3) 收集这些单独的项，并传递给collate_fn(这里是batchify函数)

3. batchify作为DataLoader原有collate_fn的替代方案，用于替换简单的张量堆叠批次生成。对于collate_fn有用的batchify部分是“padding+张量格式转换+批次形成”→定制化的批次形成方法
masks和labels的作用在于模型训练时剔除填充部分和区别正负样本。之所以在这里统一准备完成是为了保持数据处理的一致性，批量生成（掩码和标签）比每次需要时单独生成更高效，同时保持训练代码的简洁，因为数据已经准备好了。

DataLoader负责处理索引生成、多进程加载、内存管理等.它只负责将它们作为批次数据的一部分传递给模型,本身并不直接使用掩码和标签

4. num_workers参数用于设置数据加载时使用的子进程数，允许利用多核处理器进行并行处理
 
5. torch.tensor(centers).view(-1, 1) → 只对center进行重塑。(n, 1)通常用于表示每个样本的单个特征或者标签
Recall：DL模型的输入常需要输入为二维，即便是单个特征也是如此

# 3. Skip Gram Model


```python
from torch import nn

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        self.embed_v = nn.Embedding(vocab_size, embed_size)
        self.embed_u = nn.Embedding(bocab_size, embedi_size)

    def forward(self, center,contexts_and_negatives):
        return skip_gram(center, contexgts_and_nagatives, self.embed_v, self.embed_u)


if __name__ == "main":
    vocab_size = 10000
    embed_size = 100
    model = SkipGramModel(vocab_size, embed_size)

    center = torch.randint(0, vocab_size, (64, 1))
    contexts_and_negatives = torch.randingt(0, vocab_size, (64, 20))

    output = model(center, contexts_and_negatives)
    print("Output shape:", output.shape)
```

1. 维度变换类似于nn.Linear()的变体。由于skip_gram的计算逻辑更复杂，单独的模块定义使其可复用，更灵活和模块化，专注于跳字计算逻辑
2. 函数和类的定义顺序通常不会影响它们的使用，只要在调用时它们已经被定义就行
3. torch.bmm是批次矩阵乘法; permute调整矩阵维度顺序,匹配运算

# 4. Similarity Calculation


```python
def get_siilar_tokens(query_token, k, embed, idx_to_token, token_to_idx):
    W = embed.weight.data
    x = W[token_to_idx[query_token]]'
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
    _, topk = torch.topk(cos, k=k+1)
    topk = topk.cpu().numpy()
    for i in topk[1:]:  # 除去输入词
        print(f'cosine sim={cos[i]:.3f}: {idx_to_token[i]}')
    
```

添加1e-9增加数值稳定性

# 5. Training(main)


## 1) Define a class for later loss_f calling


```python
import torch
from torch import nn
import time
import sys
from DL10_3_2_preprocess import get_preprocessed_data
from DL10_3_3_dataset_sampling import load_ptb_data
from DL10_3_4_model import SkipGramModel


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, mask=None):
        inputs, tagets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", 
                                                             weight=mask)
        return res.mean(dim=1)


def train(net, data_iter, loss, num_epoch, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

for epoch in range(num_epoch):
    start, l_sum, n = time.time(), 0.0, 0
    for batch in data_iter:
        center, context_negative, mask, label = [d.to(device) for d in batch]
        pred = net(center, context_negative)
        l = loss((pred.view(label.shape), label, mask) *
                 mask.shape[1] / mask.float().sum(dim=1)).mean()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.cpu().item()
        n += 1
    print(F'EPOCH {epoch+1}, loss {l_sum / n:.4f}, time(time.time() - start:.2f}s')



```

1. 用类来定义损失函数的几种考量
    1) 损失函数逻辑较复杂/需要自定义计算 → SigmoidBinaryCrossEntropyLoss 类包含特殊的“掩码处理”和“降维操作”
    2) 损失函数计算中需要维护状态 → 训练过程中需要动态调整权重
    3) 损失计算中添加额外功能 → 日志记录、梯度检查
    4) 大型项目中保持代码结构一致性 → 便于未来extension / modification
2. reduction="none"参数表示不对结果进行降维处理 → inputs维度=res维度
word2vec模型中，模型输出(及损失函数inputs)的维度一般是:[batch_size, num_context_words, embed_size]
res.mean(dim=1) → 沿着特征维度计算平均值 

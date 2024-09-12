"""这个项目是对“跳字模型”和“负采样近似训练”的实现
同时引入一些实现中技巧,如二次采样"""


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
    file_path = os.path.join(data_dir, 'ptb.train.txt')
    
    # 如果文件已存在，直接读取数据
    if os.path.exists(file_path):
        print("PTB dataset already exists. Loading from file...")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            raw_dataset = [st.split() for st in lines]
        print('# sentences:', len(raw_dataset))
        return raw_dataset

    # 如果文件不存在，下载并处理数据
    # 创建目录
    os.makedirs(data_dir, exist_ok=True)
    
    # 下载数据
    print("Downloading Treebank dataset...")
    nltk.download('treebank')
    
    # 获取Treebank数据
    from nltk.corpus import treebank

    # 保存训练数据
    print("Saving training data...")
    with open(file_path, 'w', encoding='utf-8') as f:
        for sent in treebank.sents():
            f.write(' '.join(sent) + '\n')

    print("Download and extraction complete.")
    print("PTB dataset is ready in", data_dir)

    # 读取训练数据
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        raw_dataset = [st.split() for st in lines]

    print('# sentences:', len(raw_dataset))

    return raw_dataset


def load_ptb_data():
        dataset = download_ptb()
        print(f"Dataset loaded with {len(dataset)}sentences.")
        return dataset

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
    dataset = load_ptb_data()



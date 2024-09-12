import torch
import torch.utils.data as Data

class PTBDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])

    def __len__(self):
        return len(self.centers)

def batchify(data):
    """
    用作DataLoader的参数collate_fn: 输入是个长为batchsize的list, 
    list中的每个元素都是Dataset类调用__getitem__得到的结果
    """
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers.append(center)
        contexts_negatives.append(context + negative + [0] * (max_len - cur_len))
        masks.append([1] * cur_len + [0] * (max_len - cur_len))
        labels.append([1] * len(context) + [0] * (max_len - len(context)))
    return (torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives),
            torch.tensor(masks), torch.tensor(labels))

def load_ptb_data(all_centers, all_contexts, all_negatives, batch_size, num_workers):
    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True,
                                collate_fn=batchify, 
                                num_workers=num_workers)
    return data_iter

if __name__ == "__main__":
    # 测试代码
    from DL10_3_2_preprocess import get_preprocessed_data
    
    centers, contexts, negatives, vocab_info = get_preprocessed_data()
    
    # 创建一个小的测试数据集
    test_centers = centers[:1000]
    test_contexts = contexts[:1000]
    test_negatives = negatives[:1000]
    
    data_iter = load_ptb_data(test_centers, test_contexts, test_negatives, batch_size=4, num_workers=0)
    
    for batch in data_iter:
        print("Center shape:", batch[0].shape)
        print("Contexts and negatives shape:", batch[1].shape)
        print("Mask shape:", batch[2].shape)
        print("Label shape:", batch[3].shape)
        break  # 只打印第一个批次的信息

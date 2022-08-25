import torch
from datasets import load_from_disk
import random
import copy
import pickle

#句子合并
def get_new_sentence(item, left_lenth, is_first_sentence):
    if is_first_sentence:
        if isinstance(item,list):
            print(item)
            new_sentence_id=[0]+item
        else:
            new_sentence_id=[0]+item['input_ids']
        if len(new_sentence_id)>left_lenth-1: # trunction
            new_sentence_id=new_sentence_id[:left_lenth-1]
        new_sentence_id=new_sentence_id+[2]
        accept=True
    else:
        if isinstance(item,list):
            print(2,item)
            new_sentence_id=item+[2]
        else:
            new_sentence_id=item['input_ids']+[2]
        if len(new_sentence_id)>left_lenth:
            new_sentence_id=None
            accept=False
        else:
            accept=True
    if accept:
        left_lenth-=len(new_sentence_id)
    return new_sentence_id, accept, left_lenth

#把单句数据集合并成多句的list
def get_full_sentences_list(dataset, max_lenth):
    start_index=0
    left_lenth=max_lenth
    current_sentence_id=[]
    full_sentences_list=[]
    valid_lens=[]
    sentence_tail=[0]
    for i, item in enumerate(dataset):
        if left_lenth==max_lenth:
            is_first_sentence=True
        else:
            is_first_sentence=False
        new_sentence_id, accept, left_lenth=get_new_sentence(item, left_lenth, is_first_sentence)
        if accept:
            current_sentence_id.extend(new_sentence_id)
        else:
            full_sentences_list.append(copy.deepcopy(current_sentence_id))
            left_lenth=max_lenth
            valid_lens.append(len(current_sentence_id))
            current_sentence_id, accept, left_lenth=get_new_sentence(item, left_lenth, True)
            sentence_tail.append(i)
        if i>100000:
            break
    return full_sentences_list, valid_lens, sentence_tail

#替换
def replace_mlm_tokens(sentence, candidate_pred_positions, num_mlm_preds):
    # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    mlm_input_tokens = [token for token in sentence]
    pred_positions_and_labels = []
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的时间：将词替换为“<mask>”词元
        if random.random() < 0.8:
            masked_token = 50264
        else:
            # 10%的时间：保持词不变
            if random.random() < 0.5:
                masked_token = sentence[mlm_pred_position]
            # 10%的时间：用随机词替换该词
            else:
                masked_token = random.randint(0, 50263)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, sentence[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels

#以mask_rate的比例来mask句子
def dynamic_mask(sentence, mask_rate):
    #选出替换的位置
    candidate_pred_positions = []
    for i, token in enumerate(sentence):
        if token in [0, 2]:
            continue
        candidate_pred_positions.append(i)
    num_mlm_preds = max(1, round(len(sentence) * mask_rate))
    #替换
    X_input, pred_positions_and_labels = replace_mlm_tokens(
        sentence, candidate_pred_positions, num_mlm_preds)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    Y_label = [v[1] for v in pred_positions_and_labels]
    return X_input, pred_positions, Y_label

class Roberta_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2, max_lenth, mask_rate) -> None:
        self.full_sentences_list, self.valid_lens, self.sentence_tail=get_full_sentences_list(dataset1, max_lenth)
        list2, lens2, tail2=get_full_sentences_list(dataset2, max_lenth)
        self.full_sentences_list.extend(list2)
        self.valid_lens.extend(lens2)
        self.sentence_tail.extend(tail2)
        self.max_lenth=max_lenth
        self.mask_rate=mask_rate
        if mask_rate>=1:
            raise ValueError("mask的比例应该小于 1 ")
        self.max_num_mlm_preds = round(self.max_lenth * mask_rate)

    def __getitem__(self, idx):
        sentence=self.full_sentences_list[idx]
        valid_len=self.valid_lens[idx]
        X_input, pred_positions, Y_label=dynamic_mask(sentence, self.mask_rate)
        pred_len=len(pred_positions)
        #padding
        X_input=torch.tensor(X_input+[1]*(self.max_lenth-valid_len),dtype=torch.long)
        pred_positions=torch.tensor(pred_positions+[0]*(self.max_num_mlm_preds-len(pred_positions)),dtype=torch.long)
        weights=torch.tensor([1.0]*pred_len+[0.0]*(self.max_num_mlm_preds-pred_len),dtype=torch.float32)
        Y_label=torch.tensor(Y_label+[0]*(self.max_num_mlm_preds-pred_len), dtype=torch.long)
        if len(X_input)!=self.max_lenth:
            print("max_lenth error: ", X_input)
        return (X_input, valid_len, pred_positions, weights, Y_label)

    def __len__(self):
        return len(self.full_sentences_list)

def load_data(batch_size, max_len, mask_rate=0.15):
    wiki_dataset=[]
    # with open('wiki.pkl', 'rb') as f:
    #     wiki_dataset=pickle.load(f)
    bookcorpus_dataset=load_from_disk("/home/lhy/bert/roberta_dataset")
    train_set = Roberta_Dataset(wiki_dataset[:5], bookcorpus_dataset, max_len, mask_rate)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=128)
    return train_iter


if __name__=="__main__":
    batch_size, max_len = 5, 64
    train_iter = load_data(batch_size, max_len)

    for (tokens_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y) in train_iter:
        print(tokens_X)
        # print(tokens_X.shape, valid_lens_x.shape, pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,)
        break
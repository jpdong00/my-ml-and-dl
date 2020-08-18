import json
import numpy as np
from tqdm import tqdm


# 加载字典
def load_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 读取txt文件, 加载训练数据
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return [eval(i) for i in f.readlines()]


class HMM_NER:
    def __init__(self, char2idx_path, tag2idx_path):
        # 载入一些字典
        # char2idx: 字 转换为 token
        self.char2idx = load_dict(char2idx_path)
        # tag2idx: 标签转换为 token
        self.tag2idx = load_dict(tag2idx_path)
        # idx2tag: token转换为标签
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
        # 初始化隐状态数量（实体标签数）和观测数量（字典字数）
        self.tag_size = len(self.tag2idx)
        self.vocab_size = len(self.char2idx)
        # 初始化A, B, pi为全0
        self.pi = np.zeros([1, self.tag_size])
        self.transition = np.zeros([self.tag_size, self.tag_size])
        self.emission = np.zeros([self.tag_size, self.vocab_size])
        # 偏置, 用来防止log(0)或乘0的情况
        self.epsilon = 1e-8

    def fit(self, train_dic_path):
        print("Loading data...")
        train_dic = load_data(train_dic_path)
        print("Estimating pi, A and B...")
        self.estimate_parameters(train_dic)
        # 取log防止计算结果下溢
        self.pi = np.log(self.pi)
        self.transition = np.log(self.transition)
        self.emission = np.log(self.emission)
        print("DONE!")

    def estimate_parameters(self, train_dic):
        # 初始矩阵：p(i_1)
        # 转移矩阵：p(i_t+1|i_t)
        # 发射矩阵：p(o_t|i_t)
        for dic in tqdm(train_dic):
            for idx, (char, tag) in enumerate(zip(dic["text"][:-1], dic["label"][:-1])):
                cur_char = self.char2idx[char]  # 当前字在字典中的索引
                cur_tag = self.tag2idx[tag]  # 当前字的标签在标签集中的索引
                next_tag = self.tag2idx[dic["label"][idx + 1]]  # 下一个字的标签在标签集中的索引
                self.transition[cur_tag, next_tag] += 1  # 转移概率矩阵
                self.emission[cur_tag, cur_char] += 1  # 发射概率矩阵
                if idx == 0:
                    self.pi[0, cur_tag] += 1  # 初始概率矩阵
            self.emission[self.tag2idx[dic["label"][-1]], self.char2idx[dic["text"][-1]]] += 1
        # 在等于0的位置加上很小的一个值epsilon
        self.transition[self.transition == 0] = self.epsilon
        self.emission[self.emission == 0] = self.epsilon
        self.pi[self.pi == 0] = self.epsilon
        # 转移概率
        self.transition /= np.sum(self.transition, axis=1, keepdims=True)
        # 发射概率
        self.emission /= np.sum(self.emission, axis=1, keepdims=True)
        # 初始状态概率
        self.pi /= np.sum(self.pi, axis=1, keepdims=True)

    def emission_prob(self, char):
        # 计算发射概率，即p(observation|state)
        char_token = self.char2idx.get(char, 0)
        # 如果当前字属于未知, 则将p(observation|state)设为均匀分布
        if char_token == 0:
            return np.log(np.ones(self.tag_size) / self.tag_size)
        # 否则，取出发射概率矩阵char对应那一列
        else:
            return np.ravel(self.emission[:, char_token])

    def viterbi(self, text):
        # 序列长度
        seq_len = len(text)
        # 初始化T1表、T2表
        T1_table = np.zeros([self.tag_size, seq_len])
        T2_table = np.zeros([self.tag_size, seq_len])
        # 第1时刻的发射概率
        start_p_Obs_State = self.emission_prob(text[0])
        # 第一步初始概率
        T1_table[:, 0] = self.pi + start_p_Obs_State
        T2_table[:, 0] = np.nan

        for i in range(1, seq_len):
            # 当前时刻的发射概率
            p_Obs_State = self.emission_prob(text[i])
            p_Obs_State = np.expand_dims(p_Obs_State, axis=-1)  # tag_size * 1
            # 前一时刻计算出的概率值
            prev_score = np.expand_dims(T1_table[:, i - 1], axis=0)  # 1 * tag_size
            # 广播
            curr_score = prev_score + self.transition.T + p_Obs_State

            # 存入T1 T2中
            T1_table[:, i] = np.max(curr_score, axis=-1)
            T2_table[:, i] = np.argmax(curr_score, axis=-1)

        # 回溯
        best_tag_id = int(np.argmax(T1_table[:, -1]))
        best_tags = [best_tag_id]
        for i in range(seq_len - 1, 0, -1):
            best_tag_id = int(T2_table[best_tag_id, i])
            best_tags.append(best_tag_id)
        return list(reversed(best_tags))

    def predict(self, text):
        # 预测
        if len(text) == 0:
            raise NotImplementedError("输入文本为空!")
        # 维特比算法解码
        best_tag_id = self.viterbi(text)
        # 用来打印预测结果
        for char, tag_id in zip(text, best_tag_id):
            print(char + "_" + self.idx2tag[tag_id] + "|", end="")


if __name__ == '__main__':
    model = HMM_NER(char2idx_path="./corpus/char2idx.json",
                    tag2idx_path="./corpus/label2idx.json")
    model.fit("./corpus/train.txt")
    model.predict("中国周五宣布其北斗三号全球卫星导航系统正式开通，并做好为全球提供服务的准备，从而跻身由美国、俄罗斯和欧盟组成的提供空基（导航）系统的精英行列。")

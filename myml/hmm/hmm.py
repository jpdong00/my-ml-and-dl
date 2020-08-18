import numpy as np


class HMM:
    def __init__(self, pi, A, B):
        self.pi = pi  # 初始隐状态概率矩阵
        self.A = A  # 转移概率矩阵
        self.B = B  # 发射概率矩阵
        self.alpha = None
        self.beta = None

    def forward(self, O):
        T = len(O)  # 观测序列长度
        N = len(self.A)  # 状态数
        alpha = np.zeros([T, N])  # 初始化alpha矩阵

        # alpha_1(i)=p_i*b_i(o_1)
        alpha[0, :] = self.pi * self.B[:, O[0]]
        for t in range(1, T):
            # alpha_t+1(i)=\sum_j{alpha_t(j)*a_ji}*b_i(o_t+1)
            alpha[t, :] = np.sum(alpha[t - 1, :] * self.A.T, axis=1) * self.B[:, O[t]]
        # print(alpha)
        self.alpha = alpha  # shape: N * T

        return np.sum(alpha[-1, :])

    def backward(self, O):
        T = len(O)  # 观测序列长度
        N = len(self.A)  # 状态数
        beta = np.zeros([T, N])  # 初始化beta矩阵

        # beta_T=1
        beta[-1, :] = 1
        for t in range(T - 2, -1, -1):
            # beta_t(i)=\sum_j{beta_t+1(j)*a_ij*b_j(o_t+1)}
            beta[t, :] = np.sum(beta[t + 1, :] * self.A * self.B[:, O[t + 1]], axis=1)
        # print(beta)
        self.beta = beta  # shape: N * T

        # \sum_i{pi_i*b_i(o_1)*beta_1(i}
        return np.sum(self.pi * self.B[:, O[0]] * beta[0, :])

    def baumwelch(self, O, criterion=0.001):
        T = len(O)
        N = len(self.A)
        while True:
            self.forward(O)
            self.backward(O)

            # 求gamma, shape: T * N
            molecular1 = self.alpha * self.beta
            denominator1 = np.sum(molecular1, axis=1).reshape(-1, 1)
            gamma = molecular1 / denominator1

            # 求xi, shape: T * N * N
            xi = np.zeros([T - 1, N, N])
            for t in range(T - 1):
                molecular2 = self.alpha[t, :].reshape(1, -1) * self.A.T * self.B[:, O[t + 1]].reshape(-1,
                                                                                                      1) * self.beta[
                                                                                                           t + 1,
                                                                                                           :].reshape(
                    -1, 1)
                denominator2 = np.sum(molecular2)
                xi[t, :, :] = molecular2.T / denominator2

            # 递推
            newpi = gamma[0, :]
            newA = np.sum(xi, axis=0) / np.sum(gamma[:-1, :], axis=0).reshape(-1, 1)
            newB = np.zeros(self.B.shape)
            for k in range(self.B.shape[1]):
                mask = (O == k)
                newB[:, k] = np.sum(gamma[mask, :], axis=0) / np.sum(gamma, axis=0)

            # 终止
            if np.max(abs(self.pi - newpi)) < criterion and \
                    np.max(abs(self.A - newA)) < criterion and \
                    np.max(abs(self.B - newB)) < criterion:
                break

            self.A, self.B, self.pi = newA, newB, newpi

    def viterbi(self, O):
        # 初始化T1表、T2表
        T, N = len(O), len(self.A)
        T1_table = np.zeros([N, T])
        T2_table = np.zeros([N, T])

        # 时刻1
        T1_table[:, 0] = self.pi * self.B[:, O[0]]
        T2_table[:, 0] = np.nan

        for i in range(1, T):
            # 时刻t
            curr_score = T1_table[:, i - 1].reshape(1, -1) * self.A.T * self.B[:, O[i]].reshape(-1, 1)

            # 存入T1 T2中
            T1_table[:, i] = np.max(curr_score, axis=-1)
            T2_table[:, i] = np.argmax(curr_score, axis=-1)

        # 回溯
        best_tag_id = int(np.argmax(T1_table[:, -1]))
        best_tags = [best_tag_id]
        for i in range(T - 1, 0, -1):
            best_tag_id = int(T2_table[best_tag_id, i])
            best_tags.append(best_tag_id)
        return list(reversed(best_tags))

    def generateData(self, T):
        # 根据概率，返回可能的取值
        def _getFromProbs(probs):
            return np.where(np.random.multinomial(1, probs) == 1)[0][0]

        hiddenStates = np.zeros(T, dtype=int)
        observationsStates = np.zeros(T, dtype=int)
        hiddenStates[0] = _getFromProbs(self.pi)  # 产生第一个隐状态
        observationsStates[0] = _getFromProbs(self.B[hiddenStates[0]])  # 产生第一个观测状态
        for t in range(1, T):
            hiddenStates[t] = _getFromProbs(self.A[hiddenStates[t - 1]])
            observationsStates[t] = _getFromProbs((self.B[hiddenStates[t]]))

        return hiddenStates, observationsStates


def test_fwbw():
    A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    pi = [0.2, 0.4, 0.4]

    O = [0, 1, 0]
    hmm = HMM(np.array(pi), np.array(A), np.array(B))
    fp = hmm.forward(O)
    print("前向算法求得的P(O)：", fp)
    bp = hmm.backward(O)
    print("后向算法求得的P(O)：", bp)


def test_baumwelch():
    A = [[0.6, 0.4], [0.3, 0.7]]
    B = [[0.6, 0.2, 0.2], [0.1, 0.5, 0.4]]
    pi = [0.8, 0.2]

    hmm = HMM(np.array(pi), np.array(A), np.array(B))
    # states_data = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    # observations_data = [1, 0, 0, 2, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2]
    states_data, observations_data = hmm.generateData(20)
    hmm.baumwelch(observations_data, 0.00001)
    path = hmm.viterbi(observations_data)
    print("baumwelch算法估计参数，viterbi算法得到的隐状态序列：\n", path)
    print("实际的隐状态序列：\n", list(states_data))


if __name__ == '__main__':
    test_fwbw()
    print("\n")
    test_baumwelch()

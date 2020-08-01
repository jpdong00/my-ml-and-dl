import numpy as np


class NB:
    def __init__(self, lambda_):
        self.lambda_ = lambda_  # 拉普拉斯平滑
        self.feat_val = [0, 1]  # 文本分类中每一列特征的取值
        self.py = {}
        self.pxy = {}

    def fit(self, X, y):
        N, M = X.shape
        data = np.hstack((X, y.reshape(N, 1)))

        # 统计标签中的类别及其个数，即\sum{I(y_i=c_k)}
        unique_y, counts_y = np.unique(y, return_counts=True)
        y_info = dict(zip(unique_y, counts_y))

        # 对每一个类别进行遍历
        for ck, ck_count in y_info.items():
            # 计算P(Y=c_k)
            self.py['P(Y={})'.format(ck)] = (ck_count + self.lambda_) / (N + len(unique_y) * self.lambda_)

            # 取出标签=ck的所有行
            tmp_data = data[data[:, -1] == ck]

            # 对每一个特征遍历
            for col in range(M):
                # 统计类别为ck且该列特征下每个取值的个数，即\sum{I(x_ij=a_jl,y_i=c_k)}
                unique_feat, counts_feat = np.unique(tmp_data[:, col], return_counts=True)
                feat_info = dict(zip(unique_feat, counts_feat))
                # 如果该类别下的特征的取值全相等，那也需要把其它取值也加入到feat_info中
                if len(feat_info) != len(self.feat_val):
                    for v in self.feat_val:
                        feat_info[v] = feat_info.get(v, 0)
                # 对该特征下的每一个不同取值进行遍历
                for feat_val, feat_count in feat_info.items():
                    # 计算P(X^{j}=a_{j_l}|Y=c_k)
                    self.pxy['P(X({})={}|Y={})'.format(col + 1, feat_val, ck)] = (feat_count + self.lambda_) / (
                        (ck_count + len(feat_info) * self.lambda_))

    def predict(self, x):
        res = {}
        for k, v in self.py.items():
            p = np.log(v)
            ck = k.split('=')[-1][:-1]
            for i in range(len(x)):
                # 计算P(Y=c_k)\prod{P(X^{(j)}=x^{(j)}|Y=c_{k})}
                p = p + np.log(self.pxy['P(X({})={}|Y={})'.format(i + 1, x[i], ck)])
            res[ck] = p
        # print(res)

        max_p = float('-inf')
        max_cate = float('-inf')
        for cate, p in res.items():
            if p > max_p:
                max_p = p
                max_cate = cate

        return max_cate, max_p

    def score(self, Xtest, ytest):
        c = 0
        for x, y in zip(Xtest, ytest):
            cate, p = self.predict(x)
            if int(cate) == int(y):
                c += 1
        return c / len(Xtest)


# 高斯朴素贝叶斯
class GNB:
    def __init__(self):
        self.parameters = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = list(np.unique(y))

        for c in self.classes:
            # 计算每个类别的平均值，方差，先验概率
            X_Index_c = X[np.where(y == c)]
            X_index_c_mean = np.mean(X_Index_c, axis=0, keepdims=True)
            X_index_c_var = np.var(X_Index_c, axis=0, keepdims=True)
            prior = X_Index_c.shape[0] / X.shape[0]
            self.parameters["class" + str(c)] = {"mean": X_index_c_mean, "var": X_index_c_var, "prior": prior}
        # print(self.parameters)

    def predict(self, X):
        # 取概率最大的类别返回预测值
        output = []
        for y in self.classes:
            # 先验概率
            prior = np.log(self.parameters["class" + str(y)]["prior"])

            # 后验概率：一维高斯分布的概率密度函数
            mean = self.parameters["class" + str(y)]["mean"]
            var = self.parameters["class" + str(y)]["var"]

            eps = 1e-4
            numerator = np.exp(-(X - mean) ** 2 / (2 * var + eps))
            denominator = np.sqrt(2 * np.pi * var + eps)

            # 取对数防止数值溢出
            posterior = np.sum(np.log(numerator / denominator), axis=1, keepdims=True).T
            prediction = prior + posterior
            output.append(prediction)

        output = np.reshape(output, (len(self.classes), X.shape[0]))
        prediction = np.argmax(output, axis=0)
        return prediction

    def score(self, X_test, y_test):
        pred = self.predict(X_test)
        right = (y_test - pred == 0.0).sum()

        return right / float(len(X_test))



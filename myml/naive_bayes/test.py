from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split

from naive_bayes import NB, GNB


# 统计学习方法例4.1
def example():
    d = {'S': 0, 'M': 1, 'L': 2}
    X = np.array([[1, d['S']], [1, d['M']], [1, d['M']],
                  [1, d['S']], [1, d['S']], [2, d['S']],
                  [2, d['M']], [2, d['M']], [2, d['L']],
                  [2, d['L']], [3, d['L']], [3, d['M']],
                  [3, d['M']], [3, d['L']], [3, d['L']]])
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    model = NB(lambda_=0.2)
    model.fit(X, y)
    cate, p = model.predict(np.array([2, 0]))
    print("最可能的类别: {}".format(cate))


# 数据清洗：转为小写，去掉少于2个字符的字符串
def text_parse(big_str):
    token_list = re.split(r'\W+', big_str)  # \W：匹配任何非单词字符
    if len(token_list) == 0:
        print(token_list)
    return [tok.lower() for tok in token_list if len(tok) > 2]


# 垃圾邮件检测
def spam_test():
    # 读入数据
    doc_list = []
    class_list = []
    for filename in ['ham', 'spam']:
        for i in range(1, 26):
            with open("./email/" + filename + '/' + str(i) + '.txt') as f:
                words = f.read()
                words = text_parse(words)
            doc_list.append(' '.join(words))
            if filename == 'ham':
                class_list.append(1)
            else:
                class_list.append(-1)

    # 单词计数
    vec = CountVectorizer()
    words = vec.fit_transform(doc_list)
    words = pd.DataFrame(words.toarray(), columns=vec.get_feature_names())
    # 转为二值，单词出现为0，没出现为1
    words[words > 0] = 1

    # 构建数据集
    X = words.values
    y = np.array(class_list)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

    # 训练
    model = NB(lambda_=0.2)
    model.fit(np.array(Xtrain), np.array(ytrain))

    # 测试
    score = model.score(Xtest, ytest)
    print('正确率：{}'.format(score))


def iris_test():
    iris = load_iris()  # 鸢尾花数据集
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    # 构建数据集
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # 训练
    model = GNB()
    model.fit(X_train, y_train)

    # 测试
    ck = model.predict(np.array([4.4, 3.2, 1.3, 0.2]).reshape(1, -1))
    print("预测的类别是：{}".format(ck))

    score = model.score(X_test, y_test)
    print('正确率：{}'.format(score))


if __name__ == '__main__':
    example()
    spam_test()
    iris_test()

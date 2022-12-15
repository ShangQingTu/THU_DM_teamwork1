from math import log
import random
import pickle
import os
import numpy as np

import pandas as pd

USE_ID3 = True


class DataSet():
    def __init__(self, dataSet_x, dataSet_y, col2name_x, col2name_y):
        if dataSet_y is not None:
            assert len(dataSet_x) == len(dataSet_y)
        self.x_data = np.array(dataSet_x).tolist()
        self.y_data = np.array(dataSet_y).tolist()
        self.col2name_x = col2name_x
        self.col2name_y = col2name_y

    def index_to_x(self, index):
        return self.x_data[index]

    def genSubData(self, validIndexList):
        """
        :param validIndexList: list 被选择作为valid集的数据的索引
        :return: 划分好的dataSet,包括train和valid
        """
        train_x, valid_x = [], []
        train_y, valid_y = [], []
        for i in range(len(self.x_data)):
            if i in validIndexList:
                valid_x.append(self.x_data[i])
                valid_y.append(self.y_data[i])
            else:
                train_x.append(self.x_data[i])
                train_y.append(self.y_data[i])
        sub_train = DataSet(train_x, train_y, self.col2name_x, self.col2name_y)
        sub_valid = DataSet(valid_x, valid_y, self.col2name_x, self.col2name_y)
        return sub_train, sub_valid

    def genRadomData(self, choiceList):
        valid_x = []
        valid_y = []
        for i in choiceList:
            valid_x.append(self.x_data[i])
            valid_y.append(self.y_data[i])
        radomDataSet = DataSet(valid_x, valid_y, self.col2name_x, self.col2name_y)
        return radomDataSet

    def stripeIndex(self):
        self.col2name_x = self.col2name_x[1:]
        self.col2name_y = self.col2name_y[1:]
        for i in range(len(self.x_data)):
            self.x_data[i] = self.x_data[i][1:]
            self.y_data[i] = self.y_data[i][1:]

    def stripeXIndex(self):
        self.col2name_x = self.col2name_x[1:]
        for i in range(len(self.x_data)):
            self.x_data[i] = self.x_data[i][1:]


class ClassifyModel():
    def __init__(self, tree_num):
        """
          :param tree_num:Int  树的量

          """
        self.model = RadomForest(tree_num)

    def fit(self, X_train, Y_train):
        i2name_x = []
        for i, name in enumerate(X_train.columns.values):
            i2name_x.append(name)
        # i2name_y = {0: 'label'}

        i2name_y = []
        for i, name in enumerate(Y_train.columns.values):
            i2name_y.append(name)
        dataSet = DataSet(X_train, Y_train, i2name_x, i2name_y)
        self.train(dataSet)

    def predict(self, X_test):
        i2name_x = []
        for i, name in enumerate(X_test.columns.values):
            i2name_x.append(name)
        dataSet = DataSet(X_test, None, i2name_x, None)
        predList = self.valid(dataSet)
        return predList

    def train(self, trainSet):
        """
        :param trainSet:DataSet  需要用于集合的集合的X数据

        """
        self.model.creatForest(trainSet)

    def valid(self, validSet):
        """
        :param validSet:DataSet  需要用于测试的集合的X
        :return: list 预测的结果
        """
        col2name_x = validSet.col2name_x
        x_data = validSet.x_data
        predList = [self.model.pridict(col2name_x, row) for row in x_data]

        return predList

    def save(self, save_dir, fold, f1):
        """
        :param save_dir: path 保存的路径
        :param fold: int 表示现在是交叉验证的第几折
        :param f1: int 在验证集的 f1分数
        :return: 把模型保存好
        """
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        model_file_name = "{}_fold{}_val{}.pkl".format("forest", fold, format(f1, '.3f'))
        model_save_path = os.path.join(save_dir, model_file_name)
        fout = open(model_save_path, "wb")
        pickle.dump(self, fout)
        fout.close()


class DecisionTree():
    def __init__(self, x_data, y_data, col2name_x):
        # 约定:不带index的
        self.x_data = x_data
        self.y_data = y_data
        self.col2name_x = col2name_x
        self.numEntries = len(self.y_data)
        # 特征的数量
        numFeatures = len(self.x_data[0])
        self.visible_attr_num = int(pow(numFeatures, 0.5))

    def calcShannonEnt(self, y_data, is_twoDim=True):
        numEntries = len(y_data)
        # 这里是统计类的标签　的　个数
        labelCounts = {}
        for featVec in y_data:
            if is_twoDim:
                label = featVec[-1]
            else:
                label = featVec
            if label not in labelCounts.keys():
                labelCounts[label] = 0
            labelCounts[label] += 1
        # 保存信息熵
        shannonEnt = 0.0
        for key in labelCounts:
            # 比例
            prob = float(labelCounts[key]) / numEntries
            # 熵
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

    def splitDataSet(self, axis, value):
        ret_x_data = []
        ret_y_data = []
        for i in range(len(self.y_data)):
            featVec = self.x_data[i]
            label = self.y_data[i]
            if featVec[axis] == value:
                # 不带value这列的reducedFeatVec
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                ret_x_data.append(reducedFeatVec)
                ret_y_data.append(label)

        return ret_x_data, ret_y_data

    # 选择最好的特征
    def chooseBestFeatureToSplit(self):
        # 　特征的个数
        numFeatures = len(self.x_data[0])
        # 随机从中选　m 个特征
        myChoice = [random.randint(0, numFeatures - 1) for j in range(self.visible_attr_num)]
        # 数据集的熵
        baseEntropy = self.calcShannonEnt(self.y_data)
        # 最大的增益值
        bestInfoGain = 0.0
        # 增益最大的特征
        bestFeature = -1
        # for i in range(numFeatures):
        for i in myChoice:
            featValueList = [example[i] for example in self.x_data]
            # i号特征可能的取值
            uniqueValues = set(featValueList)
            newEntrophy = 0.0
            for value in uniqueValues:
                subData_x, subData_y = self.splitDataSet(i, value)
                prob = len(subData_x) / float(self.numEntries)
                newEntrophy += prob * self.calcShannonEnt(subData_y)
            # 信息增益
            if USE_ID3:
                infoGain = baseEntropy - newEntrophy
            else:
                infoGain = baseEntropy - newEntrophy
                EntrophyHA = self.calcShannonEnt(featValueList, False)
                try:
                    infoGain = infoGain / EntrophyHA
                except Exception:
                    continue
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    def majorityCnt(self):
        positive_num = 0
        for labelVec in self.y_data:
            if labelVec[-1] == 1:
                positive_num += 1
            else:
                positive_num -= 1
        # 是负数多还是正数多
        if positive_num > 0:
            return 1
        else:
            return -1

    def creatTree(self):
        if self.y_data.count(self.y_data[0]) == len(self.y_data):
            # print("y_data[0]")
            return self.y_data[0]

        if len(self.col2name_x) == 0:
            # print("majorityCnt")
            return self.majorityCnt()

        bestFeat = self.chooseBestFeatureToSplit()
        bestFeatLabel = self.col2name_x[bestFeat]
        # 初始决策树
        myTree = {bestFeatLabel: {}}
        subLabels = self.col2name_x[:bestFeat]
        subLabels.extend(self.col2name_x[bestFeat + 1:])
        featValueList = [example[bestFeat] for example in self.x_data]
        # bestFeat号特征可能的取值
        uniqueValues = set(featValueList)
        for value in uniqueValues:
            ret_x_data, ret_y_data = self.splitDataSet(bestFeat, value)
            decisionTree = DecisionTree(ret_x_data, ret_y_data, subLabels)
            myTree[bestFeatLabel][value] = decisionTree.creatTree()
        return myTree


class RadomForest():
    def __init__(self, tree_num):
        self.tree_num = tree_num
        # attr_num 没用上, 最后还是开根号了
        # self.attr_num = attr_num
        self.trees = []

    def creatForest(self, trainSet):
        dataNum = len(trainSet.x_data)
        if self.tree_num == 1:
            # 普通的决策树
            decisionTree = self.creatTree(trainSet)
            self.trees.append(decisionTree)
            return
        #  随机产生self.tree_num棵树
        for i in range(self.tree_num):
            # 随机抽样dataNum次
            myChoice = [random.randint(0, dataNum - 1) for j in range(dataNum)]
            radomDataSet = trainSet.genRadomData(myChoice)
            decisionTree = self.creatTree(radomDataSet)
            # print(decisionTree)
            self.trees.append(decisionTree)

    def creatTree(self, trainSet):
        decisionTreeModel = DecisionTree(trainSet.x_data, trainSet.y_data, trainSet.col2name_x)
        decisionTree = decisionTreeModel.creatTree()
        return decisionTree

    def pridict_by_one_tree(self, inputTree, col2name_x, testVec):
        """
        :param inputTree:　输入的树
        :param col2name_x:　List 树的列的index　转换　名字
        :param testVec: 测试的输入数据
        :return: 长度为１的list , 是预测的标签
        """
        firstKeyList = list(inputTree.keys())
        # print("firstStr: ",firstStr)
        firstStr = firstKeyList[0]
        secondDict = inputTree[firstStr]
        # 根的特征是特征第几列
        featIndex = col2name_x.index(firstStr)
        # 取出要测试的根的值
        key = testVec[featIndex]
        try:
            valueOfFeat = secondDict[key]
        except Exception:
            return [-1]
        if isinstance(valueOfFeat, dict):
            predLabel = self.pridict_by_one_tree(valueOfFeat, col2name_x, testVec)
        else:
            predLabel = valueOfFeat
        return predLabel

    def pridict(self, col2name_x, testVec):
        resPredDic = {}
        for tree in self.trees:
            predLabel = self.pridict_by_one_tree(tree, col2name_x, testVec)[-1]
            if predLabel not in resPredDic.keys():
                resPredDic[predLabel] = 0
            resPredDic[predLabel] += 1
        pred = 1
        maxProb = -1
        for k, v in resPredDic.items():
            if v > maxProb:
                maxProb = v
                pred = k

        return pred


if __name__ == '__main__':
    #　样例
    X_train = pd.DataFrame({"feature1": [0, 1, 0], "feature2": [1, 1, 0]})
    X_test = pd.DataFrame({"feature1": [0, 1], "feature2": [1, 1]})
    Y_train = pd.DataFrame({"label": [1, 0, 1]})
    # tree_num取1时就是决策树, 取更大就是随机森林
    model = ClassifyModel(tree_num=1)
    model.fit(X_train, Y_train)
    Y_result = model.predict(X_test)
    print("predict is", Y_result)
    # predict is [1, 0]

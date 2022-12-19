from collections import defaultdict
from enum import auto
from unicodedata import name
import pandas as pd
import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

import seaborn as sns


import torch

from model.RadomForest import ClassifyModel
#用于分类模型
def main(args):
    std = StandardScaler()

    data_name = args['data_name'] 
    data = pd.read_csv('data/'+data_name + '.csv', encoding='utf-8')
    
    # data_X = np.array(data)
    data_X = pd.DataFrame(data)
    train_id = int(len(data_X)*args['train_size'])
    X_train = data_X[:train_id]
    X_test = data_X[train_id:]
    
    X_train = pd.DataFrame(std.fit_transform(X_train))
    X_test = pd.DataFrame(std.transform(X_test))

    label = pd.read_csv('data/label.csv', encoding='utf-8')
    # data_Y = np.array(label)
    data_Y = pd.DataFrame(label['label'])
    Y_train = data_Y[:train_id]
    Y_test = data_Y[train_id:]
    # import pdb;pdb.set_trace()
    if args['model_name'] == 'mlp':
        # clf = MLPClassifier(hidden_layer_sizes=(876,876,512), activation='relu', learning_rate_init=args['lr'],
        #                     max_iter=args['epoch'], momentum=args['momentum'], shuffle=True)#用于单模型实现
        grid = MLPClassifier(hidden_layer_sizes=(876,876,512), activation='relu', max_iter=args['epoch'], shuffle=True)
        param_dist = {
                'learning_rate_init':np.linspace(0.01*args['lr'],2*args['lr'],4),
                'momentum':np.linspace(0.01*args['momentum'],args['momentum'],4),
                }
        clf = RandomizedSearchCV(grid, param_dist, n_jobs = -1)
    elif args['model_name'] == 'rdf':
        clf = ClassifyModel(tree_num=args['n_estimators'])
        # clf = RandomForestClassifier(n_estimators=args['n_estimators'])
        # grid = ClassifyModel()
        # param_dist = {
        #         'n_estimators':range(int(0.5*args['n_estimators']),2*args['n_estimators'],4),
        #         }
        # clf = RandomizedSearchCV(grid, param_dist, n_jobs = -1)
    elif args['model_name'] == 'xgb':
        # clf = XGBClassifier(n_estimators=args['n_estimators'], n_jobs=-1,  )#用于单模型实现
        grid = XGBClassifier(n_estimators=args['n_estimators'], n_jobs=-1,  )
        param_dist = {
                'n_estimators':range(int(0.5*args['n_estimators']),2*args['n_estimators'],4),
                'max_depth':range(2,15,1),
                'learning_rate':np.linspace(0.01,2,20),
                }
        clf = RandomizedSearchCV(grid, param_dist, n_jobs = -1)
    elif args['model_name'] == 'svm':
        # clf = svm.SVC(C=args['c'], kernel='rbf', )#用于单模型实现
        grid = svm.SVC(C=args['c'])
        param_dist = {
                'kernel': ['rbf', 'linear', 'poly'],
                'C': [0.001, 0.01, 1, 3],
                'gamma': [0.01, 0.1, 1]
                }
        clf = RandomizedSearchCV(grid, param_dist, n_jobs = -1)
    elif args['model_name'] == 'knn':
        # clf = KNeighborsClassifier(n_neighbors=args['k'])#用于单模型实现
        grid = KNeighborsClassifier()
        param_dist = {
                "weights":["uniform", "distance"],
                "n_neighbors":range(2,15,1)
                }
        clf = RandomizedSearchCV(grid, param_dist, n_jobs = -1)
        
    clf.fit(X_train, Y_train)
    Y_result = clf.predict(X_test)
    print(accuracy_score(Y_test,Y_result), roc_auc_score(Y_test, Y_result), f1_score(Y_test, Y_result))
    import pdb;pdb.set_trace()
    print(clf.best_estimator_)
    print(clf.best_params_)
    
    # import pdb;pdb.set_trace()
    return accuracy_score(Y_test,Y_result), roc_auc_score(Y_test, Y_result), f1_score(Y_test, Y_result)
    





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                            choices=['cpu', 'gpu'])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--alias', type=str, default='test')
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--plt', type=bool, default=True)
    
    parser.add_argument('--model_name',type=str,default='rdf',
                            choices=['mlp', 'rdf', 'xgb', 'svm', 'knn'])
    parser.add_argument('--data_name',type=str,default='origin_feature',
                            choices=['origin_feature', 'feature_plus_extra'])
    parser.add_argument('--train_size', type=float, default=0.8)

    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epoch' ,type=int , default=100)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--k', type=int, default=5)


    args = parser.parse_args()
    args = vars(args)
    print(main(args))
    
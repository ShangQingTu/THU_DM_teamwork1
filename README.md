# THU_DM_teamwork1
Code for the news popularity prediction task.



# Data Analysis

- data_channel_is_lifestyle, data_channel_is_entertainment, data_channel_is_bus, data_channel_is_socmed, data_channel_is_tech, data_channel_is_world 其实是一个属性channel，mashable网站把新闻分成了6个channel
- weekday_is_monday, weekday_is_tuesday, weekday_is_wednesday, weekday_is_thursday, weekday_is_friday, weekday_is_saturday, weekday_is_sunday, is_weekend也是一个属性日期，看是星期几

我们将画图分析数据的分布 (TODO)

```
python preprocess.py --task analyse
```



# Preprocess

- 预处理，需要把下载的`OnlineNewsPopularity.csv`放到`./data`下

- 在`./data`下输出 `extra_feature.csv`,` origin_feature.cs` 和 `label.csv`
  - ` extra_feature.csv`是通过爬虫获取的额外特征，包括: author_id,year,month,hour
  - 例如：![](./figures/author_and_time.png)
  - 爬虫的时间有点慢:![](./figures/spider_cost.png)
- 原始的 61个属性中, url, timedelta是与热度无关的,去掉

```
python preprocess.py --task preprocess
```



# Model

存放了一个手写的决策树

```python
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
```

# Training

使用了手写完成的决策树分类模型（DT）以及对应机器学习库中的支持向量机（svm）、多层感知机（mlp）、xgboost和k近邻节点（knn）模型，一共五种算法。针对模型的部分可调参数使用了随机搜索算法探索了对应模型的可行最佳超参数取值，对应结果如下表所示

| 模型      | 参数 | Accuracy | AUC_score | F1_score |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| DT      |         |0.50032|0.50023|0.48851|
| MLP     | hidden_layer_sizes=(876,876,512), activation='relu', learning_rate_init=0.001337, max_iter=100,momentum=0.504        |0.47888|0.5|0.64762|
| SVM     | C=1.0, kernel='rbf'， gamma=1e-12        |0.47812|0.49904|0.64596|
| Xgboost | 'n_estimators': 5, 'max_depth': 4, 'learning_rate': 0.2195       |0.65986|0.66187| 0.66642|
| KNN     | n_neighbors=14, weights='distance'        |0.58431|0.58498|0.58055|

通过增加额外的信息——作品发布的年月日时间以及作者（id编号），针对上述机器学习模型中性能最好的xgboost额外再次进行了训练和超参数的随机搜索，对应结果如下

| 模型      | 参数 | Accuracy | AUC_score | F1_score |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Xgboost      | 'n_estimators': 5, 'max_depth': 3, 'learning_rate': 0.1147        |0.648379|0.64951|0.64816|
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


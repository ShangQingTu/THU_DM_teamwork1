import json

import pandas as pd
import requests
import chardet
import re
import argparse
import time
from tqdm import tqdm
from bs4 import BeautifulSoup

month2num = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sept': 9,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    "Dec": 12
}


class HtmlParser(object):
    def __init__(self, url):
        self.url = url
        self.html_text = self.html_download()

    def html_download(self):
        """
        用于下载网页内容
        :return: 返回html内容
        """
        if self.url is None:
            return None

        user_agent = "Mozilla/4.0 (compatible;MSIE 5.5; Windows NT)"
        # 伪装成浏览器
        headers = {"User-Agent": user_agent}
        # 获取页面
        r = requests.get(self.url, headers=headers, verify=False)
        # 如果返回的响应码为200,即可成功连接服务器
        if r.status_code == 200:
            # 自动辨别网页的编码格式
            r.encoding = chardet.detect(r.content)["encoding"]
            return r.text

        return None

    def parse(self):
        """
        用于解析网页内容，抽取作者信息
        :return: 返回作者和时间数据
        """
        if self.html_text is None:
            return None

        soup = BeautifulSoup(self.html_text, "html.parser", from_encoding="utf-8")
        # 　抽取作者
        link = soup.find('a', href=re.compile(r'/author/\w+'))
        try:
            author = link.get_text()
        except AttributeError:
            author = "nobody"
        # 　抽取时间
        t_link = soup.find('time')
        raw_str = str(t_link)
        pattern = re.compile("\".*\"")
        obj = re.search(pattern, raw_str)
        time_str = "notime"
        if obj:
            time_str = obj.group().strip("\"")
        return author, time_str


def preprocess(args):
    """
    用于预处理，需要把下载的OnlineNewsPopularity.csv放到./data下
    :return: 在./data下输出 extra_feature.csv, origin_feature.csv 和 label.csv
    """
    raw_df = pd.read_csv("./data/OnlineNewsPopularity.csv", sep=", ")
    # 　获取是否popular的label
    # print(raw_df.columns.values)
    shares = raw_df["shares"]
    labels = []
    for share in shares:
        if share >= 1400:
            labels.append(1)
        else:
            labels.append(0)
    _label_df = pd.DataFrame(labels, columns=['label'])
    label_df = pd.concat([_label_df, pd.DataFrame(shares, columns=['shares'])], axis=1, ignore_index=True)
    label_df.columns = ['label', 'shares']
    label_df.to_csv("./data/label.csv", index=False)
    # 用爬虫获取html里的一些信息
    extra_features = []
    # author2id = {}
    # old_extra_features_df = pd.read_csv("./data/extra_feature_v0.csv")
    urls = raw_df['url'][args.restart:]
    # try:
    for i, url in tqdm(enumerate(urls), total=len(urls)):
        print(url)
        html_parser = HtmlParser(url)
        try:
            author, time_str = html_parser.parse()
            # 处理author
            # try:
            #     author_id = author2id[author]
            # except KeyError:
            #     author2id[author] = len(author2id)
            #     author_id = author2id[author]
            # 处理time_str
            tokens = time_str.split()
            hour = int(tokens[-2].split(':')[0])
            month = month2num[tokens[2]]
            year = int(tokens[3])
        except TypeError:
            author = "nobody"
            year = 2013
            month = 10
            hour = 12
        extra_feature = {
            'author': author,
            'year': year,
            'month': month,
            'hour': hour,
        }
        extra_features.append(extra_feature)
        fout = open(f"./data/extra/{args.restart + i}.json", "w")
        json.dump(extra_feature, fout)
        # break
        # 防反爬
        # time.sleep(1)
    # except Exception:
    #     pass
    ef_df = pd.DataFrame(extra_features)
    ef_df.to_csv("./data/extra_feature.csv", index=False)
    # 原始的 61个属性中, url, timedelta是与热度无关的,去掉
    _df = raw_df.drop(columns=['url', 'timedelta', 'shares'])
    _df.to_csv("./data/origin_feature.csv", index=False)


def merge():
    extra_features = []
    author2id = {}
    for i in tqdm(range(39644)):
        # 读取
        extra_json_path = f"./data/extra/{i}.json"
        with open(extra_json_path, 'r') as fin:
            extra_feature = json.load(fin)
            author = extra_feature["author"]
            # 处理author
            try:
                author_id = author2id[author]
            except KeyError:
                author2id[author] = len(author2id)
                author_id = author2id[author]
            extra_feature["author_id"] = author_id
            extra_features.append(extra_feature)
    # save
    ef_df = pd.DataFrame(extra_features).drop(columns=['author'])
    ef_df.to_csv("./data/extra_feature.csv", index=False)


def analyse():
    data = pd.read_csv('data/origin_feature.csv', encoding='utf-8')
    data_X = pd.DataFrame(data)
    train_id = int(len(data_X) * 0.8)
    X_train = data_X[:train_id]
    X_test = data_X[train_id:]
    class_feature_names = []
    print(X_train.columns.values)
    for i, feature_name in tqdm(enumerate(X_train.columns.values)):
        values = []
        for v in X_train[feature_name]:
            if v not in values:
                values.append(v)
        is_class_feature = True
        for v in X_test[feature_name]:
            if v not in values:
                is_class_feature = False
                break
        if is_class_feature:
            class_feature_names.append(i)
    print(class_feature_names)
    # ['num_keywords', 'data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world', 'kw_min_min', 'weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend', 'min_positive_polarity', 'max_positive_polarity', 'max_negative_polarity']
    # [10, 11, 12, 13, 14, 15, 16, 17, 29, 30, 31, 32, 33, 34, 35, 36, 49, 50, 53]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess and analyse data for news popularity')
    parser.add_argument('--restart', help='重启的位置', default=12347)
    parser.add_argument('--task', help='任务类型', default='analyse',
                        choices=['preprocess', 'merge', 'analyse'])
    args = parser.parse_args()
    if args.task == 'preprocess':
        preprocess(args)
    elif args.task == 'merge':
        merge()
    elif args.task == 'analyse':
        analyse()

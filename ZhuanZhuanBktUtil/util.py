"""
【pypi密码】
1690992651@qq.com
username: wangyongpeng
pwd : xedpix-zopby1-jYhdin
"""
import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt

# 模型
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree._export import _BaseTreeExporter
import graphviz
from sklearn.tree import export_text
import sys
import tensorflow as tf
import pandas as pd
import toad
from sklearn.model_selection import train_test_split
from absl import flags

tf.logging.set_verbosity(tf.logging.INFO)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# 用逻辑回归建模
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score




# --------------------------------------------------------------------
# flags 传参(数据路径，标签，分桶数量)
# --------------------------------------------------------------------
# flags.DEFINE_string(name="data_path", default="./data", help="data path")
# flags.DEFINE_string(name="label", default="click", help="the label of sample")
# flags.DEFINE_integer(name="n_buckets", default=20, help="how many buckets do you want to split")
# flags.DEFINE_integer(name="epochs", default=10, help="epochs")
# flags.DEFINE_string(name="columns", default="", help="csv header columns")
# flags.DEFINE_string(name="drop_cols_str", default="", help="columns to be droped")
# flags.FLAGS(sys.argv)
# FLAGS = flags.FLAGS


# --------------------------------------------------------------------
# 清洗数据，返回dataframe
# --------------------------------------------------------------------
def build_df(data_path, columns):
    row_df = pd.read_csv(data_path, names=columns.split(","))
    # 删除没必要的列名
    drop_cols = ['ws_id', 'order_label', 'pay_label', 'sample_weight',
                 'pid', 'page', 'page_idx', 'cate_first_id', 'cate_second_id', 'cate_third_id',
                 'color_id', 'capacity_id', 'quality_id', 'net_id', 'version_id', 'sku_id',
                 'cate1id_price_level', 'cate2id_price_level', 'cate3id_price_level',
                 'cate3id_color_price_level', 'cate3id_capacity_price_level', 'cate3id_quality_price_level',
                 'cate3id_net_price_level', 'cate3id_version_price_level', 'sku_price_level',
                 'dt']
    for col in drop_cols:
        del row_df[col]

    row_df = row_df.fillna(0)
    return row_df


# --------------------------------------------------------------------
# 树分桶工具，
# 决策树版本
# @column: 特征列
# @label: 标签列
#@return： 返回特征名的分桶点
# --------------------------------------------------------------------

# 输入打印的字符串，返回分桶值
def get_bkt_from_export_text(export_text):
    bkt_list = []
    result = []
    # 先写入
    with open("tep.txt", 'w', encoding='utf-8') as w:
        w.write(export_text)
    # 再读取
    with open("tep.txt", 'r') as source:
        txt = source.readlines()
        for i in range(0, len(txt)):
            bkt_value = float(txt[i].split(" ")[-1].replace("\n", ""))
            bkt_list.append(bkt_value)
        # print(sorted(set(bkt_list)))
    return sorted(set(bkt_list))

def getFeatureBkt_DT(DF, featureCol, labelCol):
    DF = DF.fillna(0)
    # 取出单特征和标签
    x, y = DF[featureCol].values, DF[labelCol].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # 开始训练
    clf = DecisionTreeClassifier(max_depth=10)
    clf=clf.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
    # 方法1 ： plot_tree
    # tree.plot_tree(clf)
    # 方法2 ： graphviz
    # 方法3 ： export_text
    r = export_text(clf, feature_names=[featureCol]) # 返回值 str类型
    # 输入打印的结果，返回分桶值
    bkt_list = get_bkt_from_export_text(r)
    # print("'%s': %s" %(featureCol, bkt_list))
    return featureCol, bkt_list

# --------------------------------------------------------------------
# 返回决策树分桶点的字典： 包装了getFeatureBkt_DT()
# --------------------------------------------------------------------
def decission_tree_train(DF, columns, label):
    result = {}
    for col in columns:
        feature_name, bkt_list = getFeatureBkt_DT(DF, col, label)
        result[col] = bkt_list
    return result


# --------------------------------------------------------------------
# ‘step’ 用等步长的分桶结果进行LR模型的训练
# --------------------------------------------------------------------
def step_lr_train(row_df, epochs, label, n_buckets):
    auc_list = []
    bkt_disc_dict = {}
    # print("step-等步长分桶 running........")
    combiner = toad.transform.Combiner()
    for i in range(1, epochs):
        combiner.fit(row_df, y=label, method='step', n_bins=n_buckets, empty_separate=False)
        bkt_disc = combiner.export()
        # 特征变换
        x_train, x_test, y_train, y_test = rowdf2feature_transform(combiner, row_df, label)
        lr = LogisticRegression().fit(x_train, y_train)
        pred = lr.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, pred)
        # print('等步长分桶训练 epoch={} auc={}'.format(i, auc))
        auc_list.append(auc)
        key_name = "best_bkt" + str(i - 1)
        bkt_disc_dict[key_name] = bkt_disc

    # 从多个epoch中选出auc最好的
    max_index = auc_list.index(max(auc_list))
    key_name = "best_bkt" + str(max_index)
    best_bkt = bkt_disc_dict[key_name]
    return max(auc_list), best_bkt


# --------------------------------------------------------------------
# ‘quantile’ 用等频的分桶结果进行LR模型的训练
# --------------------------------------------------------------------
def quantile_lr_train(row_df, epochs, label, n_buckets):
    auc_list = []
    bkt_disc_dict = {}
    # print("quantile-等频分桶 running........")
    combiner = toad.transform.Combiner()
    for i in range(1, epochs):
        combiner.fit(row_df, y=label, method='quantile', n_bins=n_buckets, empty_separate=False)
        bkt_disc = combiner.export()
        # 特征变换
        x_train, x_test, y_train, y_test = rowdf2feature_transform(combiner, row_df, label)
        lr = LogisticRegression().fit(x_train, y_train)
        pred = lr.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, pred)
        # print('等频分桶训练 epoch={} auc={}'.format(i, auc))
        auc_list.append(auc)
        key_name = "best_bkt" + str(i - 1)
        bkt_disc_dict[key_name] = bkt_disc

    # 从多个epoch中选出auc最好的
    max_index = auc_list.index(max(auc_list))
    key_name = "best_bkt" + str(max_index)
    best_bkt = bkt_disc_dict[key_name]
    return max(auc_list), best_bkt


# --------------------------------------------------------------------
# ‘dt’ 用决策树分桶结果进行LR模型的训练
# --------------------------------------------------------------------
def dt_lr_train(row_df, epochs, label, n_buckets):
    auc_list = []
    bkt_disc_dict = {}
    # print("dt-决策树分箱 running........")
    combiner = toad.transform.Combiner()
    for i in range(1, epochs):
        combiner.fit(row_df, y=label, method='dt', n_bins=n_buckets, empty_separate=False)
        bkt_disc = combiner.export()
        # 特征变换
        x_train, x_test, y_train, y_test = rowdf2feature_transform(combiner, row_df, label)
        lr = LogisticRegression().fit(x_train, y_train)
        pred = lr.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, pred)
        # print('决策树训练 epoch={} auc={}'.format(i, auc))
        auc_list.append(auc)
        key_name = "best_bkt" + str(i - 1)
        bkt_disc_dict[key_name] = bkt_disc

    # 从多个epoch中选出auc最好的
    max_index = auc_list.index(max(auc_list))
    key_name = "best_bkt" + str(max_index)
    best_bkt = bkt_disc_dict[key_name]
    return max(auc_list), best_bkt


# --------------------------------------------------------------------
# 自定义给出多少个桶比较合适
# --------------------------------------------------------------------
def dt_best_bkt(row_df, epochs, label, n_buckets):
    auc_list = []
    bkt_disc_dict = {}
    bkt_num_list = [i for i in range(5, n_buckets)]
    combiner = toad.transform.Combiner()
    for i in bkt_num_list:
        combiner.fit(row_df, y=label, method='dt', n_bins=i, empty_separate=False)
        bkt_disc = combiner.export()
        # 特征变换
        x_train, x_test, y_train, y_test = rowdf2feature_transform(combiner, row_df, label)
        lr = LogisticRegression().fit(x_train, y_train)
        pred = lr.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, pred)
        # print('卡方正在为您选择更好的分桶数量 epoch={} auc={}'.format(i, auc))
        auc_list.append(auc)
        key_name = "best_bkt" + str(i)
        bkt_disc_dict[key_name] = bkt_disc

    # 从多个epoch中选出auc最好的
    max_index = auc_list.index(max(auc_list))
    key_name = "best_bkt" + str(bkt_num_list[max_index])
    best_bkt = bkt_disc_dict[key_name]
    return max(auc_list), best_bkt, max_index + 1


# --------------------------------------------------------------------
# ‘chi’ 用卡方分桶结果进行LR模型的训练
# --------------------------------------------------------------------

def chi_lr_train(row_df, epochs, label, n_buckets):
    auc_list = []
    bkt_disc_dict = {}
    # print("chi-卡方分桶 running........")
    combiner = toad.transform.Combiner()
    for i in range(1, epochs):
        combiner.fit(row_df, y=label, method='dt', n_bins=n_buckets, empty_separate=False)
        bkt_disc = combiner.export()
        # 特征变换
        x_train, x_test, y_train, y_test = rowdf2feature_transform(combiner, row_df, label)
        lr = LogisticRegression().fit(x_train, y_train)
        pred = lr.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, pred)
        # print('卡方分桶训练 epoch={} auc={}'.format(i, auc))
        auc_list.append(auc)
        key_name = "best_bkt" + str(i - 1)
        bkt_disc_dict[key_name] = bkt_disc

    # 从多个epoch中选出auc最好的
    max_index = auc_list.index(max(auc_list))
    key_name = "best_bkt" + str(max_index)
    best_bkt = bkt_disc_dict[key_name]
    return max(auc_list), best_bkt


# --------------------------------------------------------------------
# 卡方 - 自定义给出多少个桶比较合适
# --------------------------------------------------------------------
def chi_best_bkt(row_df, epochs, label, n_buckets):
    auc_list = []
    bkt_disc_dict = {}
    bkt_num_list = [i for i in range(5, n_buckets)]
    combiner = toad.transform.Combiner()
    for i in bkt_num_list:
        combiner.fit(row_df, y=label, method='chi', n_bins=i, empty_separate=False)
        bkt_disc = combiner.export()
        # 特征变换
        x_train, x_test, y_train, y_test = rowdf2feature_transform(combiner, row_df, label)
        lr = LogisticRegression().fit(x_train, y_train)
        pred = lr.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, pred)
        auc_list.append(auc)
        key_name = "best_bkt" + str(i)
        bkt_disc_dict[key_name] = bkt_disc

    # 从多个epoch中选出auc最好的
    max_index = auc_list.index(max(auc_list))
    key_name = "best_bkt" + str(bkt_num_list[max_index])
    best_bkt = bkt_disc_dict[key_name]
    return max(auc_list), best_bkt, max_index + 1


# --------------------------------------------------------------------
# 特征变换
# --------------------------------------------------------------------
def rowdf2feature_transform(combiner, rowdf, label):
    transer = toad.transform.WOETransformer()
    features = transer.fit_transform(combiner.transform(rowdf), rowdf[label], exclude=[label])
    # 训练测试集切分,总样本量为10W条，9w训练，1w测试
    samples = features.iloc[:100000, :]
    labels = samples.pop(label)
    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2)
    return x_train, x_test, y_train, y_test


# --------------------------------------------------------------------
# main method
# 1. 读取数据，返回处理好异常的dataframe
# method: 分箱方法，支持’chi’ (卡方分箱), ‘dt’ (决策树分箱), ‘quantile’ (等频分箱), ‘step’ (等步长分箱) --‘kmean’
# 2. 分桶，将结果存储在文件中
# 3. 将结果进行转换，准换为训练数据
# 4. 用LR进行模型训练，给出AUC最好的一个分桶
# 卡方>决策树>等步长>等频 = 0.79 > 0.76 > 0.68 >0.65
# --------------------------------------------------------------------
def run(data_path, columns, epochs, label, n_buckets,drop_cols=None):
    auc_list = []
    # 参数替换
    row_df = build_df(data_path, columns)
    # ‘step’ (等步长分箱)
    step_auc, step_disc = step_lr_train(row_df, epochs, label, n_buckets)
    auc_list.append(step_auc)

    #     # ‘quantile’ (等频分箱)
    quantile_auc, quantile_disc = quantile_lr_train(row_df, epochs, label, n_buckets)
    auc_list.append(quantile_auc)

    #     # ‘dt’ (决策树不同特征等分桶点)
    dt_auc, dt_disc = dt_lr_train(row_df, epochs, label, n_buckets)
    auc_list.append(dt_auc)

    #     # ’chi’ (卡方分箱)
    chi_auc, chi_disc = chi_lr_train(row_df, epochs, label, n_buckets)
    auc_list.append(chi_auc)

    # 决策树不同特征不同分桶点
    bkt_columns = row_df.columns
    decision_tree_disc = decission_tree_train(row_df, bkt_columns, label)

    # 返回数组中AUC最大的值的索引
    max_auc_index = auc_list.index(max(auc_list))

    if max_auc_index == 0:
        print('模型经过计算评估，推荐您等步长分桶.\n auc={} 分桶点={}'.format(step_auc, step_disc))

    if max_auc_index == 1:
        print('模型经过计算评估，推荐您等频分桶.\n auc={} 分桶点={}'.format(quantile_auc, quantile_disc))

    if max_auc_index == 2:
        dt_auc, dt_disc, best_bkt_num = dt_best_bkt(row_df, epochs, label, n_buckets)
        print('模型经过计算评估，推荐您决策树分桶.\n auc={},分桶点={}'.format(dt_auc, dt_disc))

    if max_auc_index == 3:
        chi_auc, chi_disc, best_bkt_num = chi_best_bkt(row_df, epochs, label, n_buckets)
        print('模型经过计算评估，推荐卡方分桶.\n auc={},分桶点={}'.format(chi_auc, chi_disc))

    print("********************以下是五种分桶方式的分桶结果********************")
    print('1. 等频分桶.\n 分桶点={}'.format( quantile_disc))
    print('2. 等步长分桶.\n 分桶点={}'.format( step_disc))
    print('3. 决策树等分桶点分桶.\n 分桶点={}'.format(dt_disc))
    print('4. 决策树不同特征不停分桶点分桶.\n 分桶点={}'.format(decision_tree_disc))
    print('5. 卡方分桶.\n 分桶点={}'.format(chi_disc))





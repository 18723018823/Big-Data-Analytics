import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取数据
data = pd.read_csv('电商消费数据.csv')
print(data)
data['实付金额'] = data['实付金额'].abs()
data['消费金额'] = data.groupby('用户ID')['实付金额'].transform('sum')
data['购买数量'] = data.groupby('用户ID')['购买数量'].transform('sum')
data['最后付款日期'] = data['付款日期'].apply(lambda x: 1 if pd.to_datetime(x) >= pd.to_datetime('2022-02-01') else 0)
# RFM模型实现
def rfm_model(data):
    # 计算R（最近一次购买时间）
    data['R'] = data['付款日期'].apply(lambda x: 1 if pd.to_datetime(x) >= pd.to_datetime('2022-02-01') else 0)
    # 计算F（购买频率）
    data['F'] = data.groupby('用户ID')['订单ID'].transform('size').reset_index(level=0, drop=True)
    # 计算M（购买金额）
    data['M'] = data['实付金额'].apply(lambda x: x if x > 0 else 0)
    # 计算RFM值
    data['RFM'] = data[['R', 'F', 'M']].apply(lambda row: row[0] * 10 + row[1] * 5 + row[2], axis=1)
    # 根据RFM值进行客户价值分析
    data['客户价值'] = data.groupby('用户ID')['RFM'].transform('sum').reset_index(level=0, drop=True)
    return data
# 运行RFM模型
result = rfm_model(data)
# 输出结果
print(result)
from sklearn.cluster import KMeans
# 提取RFM值作为特征
rfm_values = result[['R', 'F', 'M']].values
# 确定聚类数目
num_clusters = 3
# 应用K-means算法进行聚类
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(rfm_values)
# 获取聚类标签
cluster_labels = kmeans.labels_
# 将聚类标签添加到原始数据中
result['Cluster'] = cluster_labels
# 计算用户南北
data['用户南北'] = data['省份'].apply(
    lambda x: '0' if x in ['浙江省', '上海', '湖北省', '湖南省', '江西省', '福建省', '云南省', '贵州省', '四川省', '重庆', '广西省',
                              '广东省', '海南省', '香港', '澳门'] else '1')
# 计算用户购买频率
data['用户购买频率'] = data.groupby('用户ID')['订单ID'].transform('size').reset_index(level=0, drop=True)
print(data)
# 计算客户价值
data['客户价值'] = result['Cluster']
data['使用状态'].fillna(0, inplace=True)


def determine_threshold(data, column_name):

    # 计算数据的均值和中位数
    mean = data[column_name].mean()
    median = data[column_name].median()

    # 计算标准差
    std = data[column_name].std()

    # 设定阈值为均值和中位数的差值加上一个标准差
    threshold = mean - median + std

    return threshold

threshold1 = determine_threshold(data,'用户购买频率')
threshold2 = determine_threshold(data, '消费金额')
threshold3 = determine_threshold(data, '最后付款日期')
threshold4 = determine_threshold(data, '购买数量')
threshold5 = determine_threshold(data, '客户价值')
threshold6 = determine_threshold(data, '使用状态')
print(threshold1,threshold2,threshold3,threshold4,threshold5,threshold6)
# 设定阈值
thresholds = {
    "使用状态": threshold6,
    "消费金额": threshold2-50,
    "客户价值": threshold5,
    "最后付款日期":threshold3,
    "用户购买频率":threshold1,
}

# 制定决策规则
def make_decision(row):
    for col, threshold in thresholds.items():
        if row[col] < threshold:
            return False
    return True

# 应用决策规则到数据集上
data["decision"] = data.apply(make_decision, axis=1)

# 添加标识列
data["is_issued"] = data["decision"].apply(lambda x: 1 if x else 0)
counts = data['is_issued'].value_counts()
print(counts[1],counts[0])
# 准备饼图数据
labels = ['发', '不发']
sizes = [counts[1], counts[0]]

# 制作饼图
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # 确保饼图是圆形的

plt.title('发与不发')
plt.show()

new_features = data[['用户ID', '用户南北', '用户购买频率', '消费金额','最后付款日期','购买数量', '客户价值','使用状态','is_issued']]

new_features.to_csv('用户特征.csv', index=False)
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取数据
data = pd.read_csv('电商消费数据.csv')

data['实付金额'] = data['实付金额'].abs()
# RFM模型实现
def rfm_model(data):
    # 计算R（最近一次购买时间）
    data['R'] = data['付款日期'].apply(lambda x: 1 if pd.to_datetime(x) >= pd.to_datetime('2022-03-01') else 0)

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

# 输出结果
print(result)

labels = ['低价值客户', '中价值客户', '高价值客户']
label_dict = dict(zip(range(len(labels)), labels))

result['客户标签'] = result['Cluster'].map(lambda x: label_dict[x])
print("客户标签:")
print(result['客户标签'])
# 根据客户价值制定相应的营销策略和服务策略
strategy = {
    '低价值客户': '推送优惠券和促销活动',
    '中价值客户': '提供定制化服务和积分奖励',
    '高价值客户': 'VIP专属服务和优先体验'
}
result['营销策略'] = result['客户标签'].map(strategy)
print("营销策略:")
print(result['营销策略'])
result_1 = result[['用户ID', 'R', 'F', 'M','客户标签','营销策略']]
result_1.to_csv("客户标签.csv")

# 这里我们要分析购买数量与实付金额的关系
corr = result[['购买数量', '实付金额']].corr()
print("购买数量与实付金额的相关系数:", corr)

# 构建客户价值分析模型
group_stats = result.groupby('Cluster').agg({
    '实付金额': 'sum',
    '购买数量': 'sum'
}).reset_index()
group_stats.columns = ['Cluster', '总实付金额', '总购买数量']
print("客户群体价值:")
print(group_stats)

# 绘制用户画像 - 柱状图
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
group_stats.plot(kind='bar', x='Cluster', y='总实付金额', ax=ax[0])
ax[0].set_title('不同客户群体的总实付金额')
group_stats.plot(kind='bar', x='Cluster', y='总购买数量', ax=ax[1])
ax[1].set_title('不同客户群体的总购买数量')
plt.show()

# 绘制用户画像 - 饼图
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.pie(group_stats['总实付金额'].values, labels=group_stats['Cluster'], autopct='%1.1f%%')
ax.axis('equal')
plt.show()


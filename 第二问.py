import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# 读取数据
df = pd.read_csv('电商消费数据.csv')
df['实付金额'] = df['实付金额'].abs()
# 数据预处理
# 填充缺失值
df.fillna(0.5, inplace=True)
# 特征构造
# 用户特征
user_features = df.groupby('用户ID').agg({
    '付款日期': 'max',  # 最近一次购买日期
    '购买数量': 'sum',  # 购买数量总和
    '实付金额': 'mean',  # 平均实付金额

}).reset_index()
user_features.columns = ['用户ID', 'last_purchase_date', 'total_purchased', 'avg_actual_amount']

# 商户特征
merchant_features = df.groupby('商家ID').agg({
    '实付金额': 'sum',  # 商户实付金额总和
    '邮费': 'sum',  # 商户邮费总和
    '购买数量': 'count',  # 购买数量总数
}).reset_index()
merchant_features.columns = ['商家ID', 'total_actual_amount', 'total_postage', 'total_sales']

# 优惠券特征
coupon_features = df.groupby('使用状态').agg({
    '实付金额': 'sum',  # 优惠券实付金额总和
    '购买数量': 'count',  # 优惠券使用数量
}).reset_index()
coupon_features.columns = ['使用状态', 'total_discounted_amount', 'usage_count']

# 将特征合并到原始数据中
df = pd.merge(df, user_features, on='用户ID')
df = pd.merge(df, merchant_features, on='商家ID')
df = pd.merge(df, coupon_features, on='使用状态')

# 数据标准化
scaler = MinMaxScaler()
df['实付金额'] = scaler.fit_transform(df['实付金额'].values.reshape(-1, 1))
df['邮费'] = scaler.fit_transform(df['邮费'].values.reshape(-1, 1))

# 输出结果
print(df)
df.to_csv('特征.csv', index=False)
user_features.to_csv('user_features.csv', index=False)
merchant_features.to_csv('merchant_features.csv', index=False)
coupon_features.to_csv('coupon_features.csv', index=False)
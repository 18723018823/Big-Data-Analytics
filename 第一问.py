# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取.xlsx文件
df = pd.read_excel('data.xlsx')
# 将DataFrame保存为.csv文件
df.to_csv('电商消费数据.csv', index=False)
# 读取数据
df = pd.read_csv('电商消费数据.csv')
df = df.fillna(0.5)
df['实付金额'] = df['实付金额'].abs()
# 对使用状态字段进行整理
df['使用状态'] = df['使用状态'].replace(
    {1.0: '已使用', 0.0: '未使用', 0.5: '未领取'})
print(df['使用状态'])
# 用户消费行为概览
total_orders = df['订单ID'].count()
total_users = df['用户ID'].nunique()
total_merchants = df['商家ID'].nunique()
total_amount = df['实付金额'].sum()
average_amount = total_amount / total_orders
print(f"总订单数: {total_orders}")
print(f"总用户数: {total_users}")
print(f"总商家数: {total_merchants}")
print(f"总消费金额: {total_amount} 元")
print(f"平均消费金额: {average_amount} 元")
# 优惠券使用情况分析
coupon_received_rate = df[df['使用状态'] != '未领取']['订单ID'].count() / total_orders
coupon_used_rate = df[df['使用状态'] == '已使用']['订单ID'].count() / df[df['使用状态'] != '未使用']['订单ID'].count()
print(f"优惠券领取率: {coupon_received_rate}")
print(f"优惠券使用率: {coupon_used_rate}")
# 用户购买行为分析
user_order_count = df['用户ID'].value_counts()
user_amount_spent = df.groupby('用户ID')['实付金额'].sum()
# 地域消费行为分析
orders_by_province = df['省份'].value_counts()
orders_by_city = df['城市'].value_counts()
amount_by_province = df.groupby('省份')['实付金额'].sum()
amount_by_city = df.groupby('城市')['实付金额'].sum()
# 商家销售行为分析
merchant_order_count = df['商家ID'].value_counts()
merchant_sales_amount = df.groupby('商家ID')['实付金额'].sum()
print(amount_by_city)
print(amount_by_province)
print(merchant_sales_amount)
# 数据可视化
plt.figure(figsize=(12, 6))
sns.histplot(user_order_count, kde=True)
plt.title('用户购买频次分布')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(user_amount_spent, kde=True)
plt.title('用户购买金额分布')
plt.show()

# 省份级别的可视化
plt.figure(figsize=(12, 6))
sns.countplot(df['省份'])
plt.title('订单按省份分布')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='省份', y='实付金额', data=df)
plt.title('各省消费金额分布')
plt.show()

# 城市级别的可视化
plt.figure(figsize=(12, 6))
sns.countplot(df['城市'])
plt.title('订单按城市分布')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='城市', y='实付金额', data=df)
plt.title('各城市消费金额分布')
plt.show()

df_same_user = df.groupby('用户ID').filter(lambda x: len(x) > 1)

# 提取日期列
date_column = df['付款日期']

# 提取用户ID列
user_id_column = df['用户ID']

# 提取日期部分
df['new_column'] = date_column.str.split(' ').str[0]
new_column=df['new_column']
count_per_day = df.groupby(new_column)['用户ID'].nunique()

print(count_per_day)
plt.figure(figsize=(10, 6))
count_per_day.plot(kind='bar')
plt.title('每个日期支付的用户数统计')
plt.xlabel('日期')
plt.ylabel('用户数')
plt.show()

plt.bar(merchant_order_count.index, merchant_order_count.values, label='订单数量')
plt.bar(merchant_sales_amount.index, merchant_sales_amount.values, bottom=merchant_order_count.values, label='销售金额')

# 添加标题和标签
plt.title('商家销售行为可视化')
plt.xlabel('商家ID')
plt.ylabel('数量/金额')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(merchant_order_count.index, merchant_order_count.values)
plt.title('商家订单数量分布')
plt.xlabel('商家ID')
plt.ylabel('订单数量')
plt.show()

# 绘制商家销售金额柱状图
plt.figure(figsize=(10, 5))
plt.bar(merchant_sales_amount.index, merchant_sales_amount.values)
plt.title('商家销售金额分布')
plt.xlabel('商家ID')
plt.ylabel('销售金额')
plt.show()
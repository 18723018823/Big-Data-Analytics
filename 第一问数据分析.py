import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('电商消费数据.csv')  # 读取数据
df = df.fillna(0.5)
df['实付金额'] = df['实付金额'].abs()
def fenxi(p):
    column_data = df['%s'%p]  # 选择要分析的列
# 描述性统计
    print(column_data.describe())  # 显示均值、标准差、四分位数等
    print("Mode:", column_data.mode().values[0])  # 众数

# 数据可视化
    plt.figure(figsize=(10, 5))  # 设置图形大小

# 直方图
    plt.subplot(1, 2, 1)  # 1行2列的第一个图
    sns.histplot(column_data, kde=True)  # 显示直方图和核密度估计
    plt.title('Histogram')

# 箱线图
    plt.subplot(1, 2, 2)  # 1行2列的第二个图
    sns.boxplot(column_data)
    plt.title('Boxplot')

    plt.show()  # 显示图形

# 正态性检验 - Shapiro-Wilk检验
    shapiro_test = stats.shapiro(column_data)
    print(f'Shapiro-Wilk Test for Normality: W={shapiro_test[0]}, p-value={shapiro_test[1]}')

# 异常值检测 - IQR方法
    Q1 = column_data.quantile(0.25)  # 第一四分位数 (25th percentile)
    Q3 = column_data.quantile(0.75)  # 第三四分位数 (75th percentile)
    IQR = Q3 - Q1  # 四分位距
    print('IQR:', IQR)
    print('Lower fence:', Q1 - 1.5 * IQR)  # 下界
    print('Upper fence:', Q3 + 1.5 * IQR)  # 上界
fenxi('实付金额')
fenxi('购买数量')
fenxi('使用状态')
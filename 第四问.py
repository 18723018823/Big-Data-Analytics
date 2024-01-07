import pandas as pd
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 加载数据
data = pd.read_csv('用户特征.csv')
y = data['is_issued']
X = data[[ '用户南北', '用户购买频率', '消费金额', '最后付款日期', '购买数量', '客户价值','使用状态']]
# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# 定义逻辑回归模型
model = LogisticRegression(random_state=42)
# 训练模型
model.fit(X_train, y_train)
with open('逻辑回归.pkl', 'wb') as f:
    pickle.dump(model, f)
# 预测测试集结果
y_pred_1 = model.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred_1)
print("Accuracy1:", accuracy)
# 创建决策树分类器
model_2 = DecisionTreeClassifier()
# 训练模型
model_2.fit(X_train, y_train)
with open('决策分类.pkl', 'wb') as f:
    pickle.dump(model_2, f)
# 预测测试集
y_pred_3 = model_2.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred_3)
print('Accuracy3: %.2f' % accuracy)
# 创建随机森林分类器
model_3 = RandomForestClassifier(n_estimators=100)
# 训练模型
model_3.fit(X_train, y_train)
with open('随机森林.pkl', 'wb') as f:
    pickle.dump(model_3, f)
# 预测测试集
y_pred_4 = model_3.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred_4)
print('Accuracy4: %.2f' % accuracy)
# 创建AdaBoost分类器
model_4= AdaBoostClassifier(n_estimators=100)
# 训练模型
model_4.fit(X_train, y_train)
with open('AdaBoost.pkl', 'wb') as f:
    pickle.dump(model_4, f)
# 预测测试集
y_pred_5 = model_4.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred_5)
print('Accuracy5: %.2f' % accuracy)






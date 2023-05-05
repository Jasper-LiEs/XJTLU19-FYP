from hmmlearn.hmm import CategoricalHMM
import numpy as np
import pandas as pd
import pickle
import json
import hmmlearn.hmm as hmm
import re
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

with open(r'C:\Users\zhichen.liang19\Desktop\honey_lrx.log') as f:
    logs = [json.loads(line) for line in f]
df = pd.DataFrame(logs)
df.drop(columns='stream', axis=1, inplace=True)

ip_regex = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
transport_regex = r"\[(\w+),"
mess_regex = r"\[[^]]*\] (.*)\n"

def extract_transport(log):
    match = re.search(transport_regex, log)
    if match:
        transport = match.group(1)
        if transport.startswith("CowrieTelnet"):
            return "telnet"
        elif transport.startswith("CowrieSSH"):
            return "ssh"
    return ""

def extract_ip(log):
    match = re.search(ip_regex, log)
    if match:
        return match.group(0)
    else:
        return ""

# 提取消息文本并创建新列
df['message'] = df['log'].str.extract(mess_regex)
df['ip'] = df['log'].apply(extract_ip)
df['transport'] = df['log'].apply(extract_transport)
df.drop(columns='log', axis=1, inplace=True)
# A1: 141.98.11.57 most common ip 2939 Cowire
# print(df['ip'].value_counts())
A1 = df[df['ip']=='141.98.11.57'].copy()
A1.drop(columns='ip', axis=1, inplace=True)
A1.drop(columns='transport', axis=1, inplace=True)
A1 = A1.reset_index()
A1.drop(columns='index', axis=1, inplace=True)
A1.drop(columns='time', axis=1, inplace=True)
A1 = A1.drop(df.index[:2])
A1 = A1.reset_index()
A1.drop(columns='index', axis=1, inplace=True)

df1 = pd.DataFrame([[0, 0, 1], [1, 1, 0]], columns=['attack', 'state', 'label'])
df2 = pd.DataFrame([[0, 0, 1], [1, 1, 2], [2, 2, 3], [2, 3, 0]], columns=['attack', 'state', 'label'])
result = pd.DataFrame(columns=['attack', 'state', 'label'])
for i in range(len(A1)):
    if A1.loc[i,'message'] == 'Connection lost after 2 seconds':
        result = result.append(df1)
    elif A1.loc[i,'message'] == 'Connection lost after 1 seconds':
        result = result.append(df2)
result = result.reset_index()
result.drop(columns='index', axis=1, inplace=True)

result=result['label']

states = [0,1,2,3]##隐藏状态
n_states = len(states)##隐藏状态长度
train_size = int(len(result)*0.8)
train_data = result[:train_size]
test_data = result[train_size:]

model = CategoricalHMM(n_components=n_states, n_iter=50, tol=0.0001, verbose=True, params='te',init_params='et')
model.startprob_= np.array([1.,0,0,0])
model.n_features = 4
# model.transmat_ = np.array([
#     [0.2, 0.2, 0.2, 0.2, 0.2],
#     [0.2, 0.2, 0.2, 0.2, 0.2],
#     [0.2, 0.2, 0.2, 0.2, 0.2],
#     [0.2, 0.2, 0.2, 0.2, 0.2],
#     [0.2, 0.2, 0.2, 0.2, 0.2],
#     ])
# model.emissionprob_ = np.array([
#     [1,0,0,0,0],
#     [0,1,0,0,0],
#     [0,0,1,0,0],
#     [0,0,0,1,0],
#     [0,0,0,0,1],
# ])

X = train_data.values.reshape(-1,1)
X = X.astype(int)
# model.fit(X)
x_test = test_data.values.reshape(-1,1)
x_test = x_test.astype(int)

final = []
for i in range(100):
    model.fit(X)
    pre_list = model.predict(x_test).tolist()
    pre_list_2 = [model.transmat_[i].tolist().index(max(model.transmat_[i])) for i in pre_list]
    ans = x_test.T[0].tolist()
    count = 0
    for i in range(len(pre_list_2)):
        if pre_list_2[i] == ans[i]:
            count += 1
    final.append(count / len(pre_list_2))

print(sum(final)/len(final))
#
# import math
# print(model.startprob_)
# print(model.transmat_)
# print(model.emissionprob_)
# print(math.exp(model.score(X)))
#
# #保存模型
# output = open(r'C:\Users\zhichen.liang19\Desktop\ceshiyong1.pkl', 'wb')
# s = pickle.dump(model, output)
# output.close()
#
# # # 调用模型
# # input = open(r'C:\Users\zhichen.liang19\Desktop\data02_1000.pkl', 'rb')
# # model = pickle.load(input)
# # input.close()
#
# pre_list = model.predict(x_test.values).tolist()
# pre_list_2 = [model.transmat_[i].tolist().index(max(model.transmat_[i])) for i in pre_list]
# ans = x_test.values.T[0].tolist()
#
# count=0
# for i in range(len(pre_list_2)):
#     if pre_list_2[i] == ans[i]:
#         count+=1
# print(count/len(pre_list_2))
#
# import matplotlib.pyplot as plt
#
# y_act = ans
# y_pre = pre_list_2
# plt.plot(np.random.rand(10))
# # plt.plot(y_act)
# # plt.plot(y_pre,linestyle="-",marker="None",color='r')
# # plt.legend(['Actual', 'Predicted'])
# # plt.show()








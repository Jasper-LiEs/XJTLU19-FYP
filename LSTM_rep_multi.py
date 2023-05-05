import json
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

window_size = 2

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

# print(df['ip'].value_counts())

# A1: 141.98.11.57 most common ip 2939 Cowire
A1 = df[df['ip']=='141.98.11.57'].copy()
A1.drop(columns='ip', axis=1, inplace=True)
A1.drop(columns='transport', axis=1, inplace=True)
A1 = A1.reset_index()
A1.drop(columns='index', axis=1, inplace=True)
A1.drop(columns='time', axis=1, inplace=True)
# delete first connection
A1 = A1.drop(df.index[:2])
A1 = A1.reset_index()
A1.drop(columns='index', axis=1, inplace=True)

# A2: 128.110.156.4
A2 = df[df['ip']=='128.110.156.4'].copy()
A2.drop(columns='ip', axis=1, inplace=True)
A2.drop(columns='transport', axis=1, inplace=True)
A2.drop(columns='time', axis=1, inplace=True)
A2 = A2.reset_index()
A2.drop(columns='index', axis=1, inplace=True)

# A3: 163.172.60.130
A3 = df[df['ip']=='163.172.60.130'].copy()
A3.drop(columns='ip', axis=1, inplace=True)
A3.drop(columns='transport', axis=1, inplace=True)
A3.drop(columns='time', axis=1, inplace=True)
A3 = A3.reset_index()
A3.drop(columns='index', axis=1, inplace=True)

# A4: 159.65.154.184
A4 = df[df['ip']=='159.65.154.184'].copy()
A4.drop(columns='ip', axis=1, inplace=True)
A4.drop(columns='transport', axis=1, inplace=True)
A4.drop(columns='time', axis=1, inplace=True)
A4 = A4.reset_index()
A4.drop(columns='index', axis=1, inplace=True)

# basic component
# attack: 0=None, 1=guess, 2=url, 3=deploy, 4=run, 5=scan, 6=modify
# state: 0=normal, 1=guess, 2=compromise
# label: next state

# guess fail
df1 = pd.DataFrame([[0, 0, 1], [1, 1, 0]], columns=['attack', 'state', 'label'])
# A1 guess success
df2 = pd.DataFrame([[0, 0, 1], [1, 1, 2], [2, 2, 2], [3, 2, 0]], columns=['attack', 'state', 'label'])
# A2 guess success
df3 = pd.DataFrame([[0, 0, 1], [1, 1, 0], [4, 0, 0]], columns=['attack', 'state', 'label'])
# A3 guess success
df4 = pd.DataFrame([[0, 0, 1], [1, 1, 2], [5, 2, 2], [6, 2, 0]], columns=['attack', 'state', 'label'])
# A4 guess success
df5 = pd.DataFrame([[0, 0, 1], [1, 1, 2], [6, 2, 2], [6, 2, 0]], columns=['attack', 'state', 'label'])

result_1 = pd.DataFrame(columns=['attack', 'state', 'label'])
result_2 = pd.DataFrame(columns=['attack', 'state', 'label'])
result_3 = pd.DataFrame(columns=['attack', 'state', 'label'])

# build dataset A1
for i in range(len(A1)):
    if A1.loc[i,'message'] == 'Connection lost after 2 seconds':
        result_1 = result_1.append(df1)
    elif A1.loc[i,'message'] == 'Connection lost after 1 seconds':
        result_1 = result_1.append(df2)
# build dataset A2
for i in range(len(A2)):
    if A2.loc[i, 'message'] == 'avatar root logging out':
        result_2 = result_2.append(df3)
    elif A2.loc[i, 'message'] == '"Got remote error, code 11 reason: b''"':
        result_2 = result_2.append(df1)
# build dataset A3
for i in range(len(A3)):
    if A3.loc[i, 'message'] == 'CMD: cat /proc/cpuinfo | grep model | grep name | wc -l':
        result_3 = result_3.append(df4)
    elif A3.loc[i, 'message'] == '"Got remote error, code 11 reason: b\'Bye Bye\'"':
        result_3 = result_3.append(df1)

result_1 = result_1.reset_index()
result_1.drop(columns='index', axis=1, inplace=True)
result_1 = result_1.astype(float)
result_2 = result_2.reset_index()
result_2.drop(columns='index', axis=1, inplace=True)
result_2 = result_2.astype(float)
result_3 = result_3.reset_index()
result_3.drop(columns='index', axis=1, inplace=True)
result_3 = result_3.astype(float)

# build dataset A4
test = pd.DataFrame(columns=['attack', 'state', 'label'])
for i in range(len(A4)):
    if A4.loc[i, 'message'] == '"Got remote error, code 11 reason: b\'Bye Bye\'"':
        test = test.append(df1)
    elif A4.loc[i, 'message'] == 'CMD: cd ~; chattr -ia .ssh; lockr -ia .ssh':
        test = test.append(df5)
    elif A4.loc[i, 'message'] == 'avatar root logging out':
        test = test.append(df3)
test = test.reset_index()
test.drop(columns='index', axis=1, inplace=True)
test = test.astype(float)

labels_1 = to_categorical(result_1.iloc[:-window_size]['label'])
labels_2 = to_categorical(result_2.iloc[:-window_size]['label'],num_classes=3)
labels_3 = to_categorical(result_3.iloc[:-window_size]['label'])
ans = to_categorical(test.iloc[:-window_size]['label'])

# 创建时间窗口大小为2s的数据集
data_1 = []
data_2 = []
data_3 = []
data_4 = []

for i in range(len(result_1) - window_size):
    data_1.append(result_1.iloc[i:i + window_size].values)
# 将数据集转换为3D张量
data_1 = np.array(data_1)

for i in range(len(result_2) - window_size):
    data_2.append(result_2.iloc[i:i + window_size].values)
# 将数据集转换为3D张量
data_2 = np.array(data_2)

for i in range(len(result_3) - window_size):
    data_3.append(result_3.iloc[i:i + window_size].values)
# 将数据集转换为3D张量
data_3 = np.array(data_3)

for i in range(len(test) - window_size):
    data_4.append(test.iloc[i:i + window_size].values)
# 将数据集转换为3D张量
data_4 = np.array(data_4)

# 创建模型
model = Sequential()
model.add(LSTM(128, input_shape=(window_size, 2)))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(data_1[:,:,:2], labels_1, epochs=10, batch_size=32)
model.fit(data_2[:,:,:2], labels_2, epochs=10, batch_size=32)
model.fit(data_3[:,:,:2], labels_3, epochs=10, batch_size=32)



# # 评估模型
loss, acc = model.evaluate(data_4[:,:,:2], ans, batch_size=16)
# # print(data[:,:,:2][0])
# # print((model.predict(data[:,:,:2]))[0]==labels[0])
# # print(labels[0])
# # loss, acc = model.evaluate(data[:,:,:2], labels, batch_size=32)
#
# print('Test loss:', loss)
print('Test accuracy:', acc)



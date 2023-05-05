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
# A1['message'].to_csv(r'C:\Users\zhichen.liang19\Desktop\A1.csv')
# A2 = df[df['ip']=='128.110.156.4'].copy()
# A2.drop(columns='ip', axis=1, inplace=True)
# A2.drop(columns='transport', axis=1, inplace=True)
# A2 = A2.reset_index()
# A2['message'].to_csv(r'C:\Users\zhichen.liang19\Desktop\A2.csv')
#
# A3 = df[df['ip']=='163.172.60.130'].copy()
# A3.drop(columns='ip', axis=1, inplace=True)
# A3.drop(columns='transport', axis=1, inplace=True)
# A3 = A3.reset_index()
# A3['message'].to_csv(r'C:\Users\zhichen.liang19\Desktop\A3.csv')
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
# print(result.shape)
# print((A1['message'].value_counts()['Connection lost after 2 seconds'])*2+(A1['message'].value_counts()['Connection lost after 1 seconds'])*4)

result = result.astype(float)

# 将标签进行独热编码
labels = to_categorical(result.iloc[:-2]['label'])
# 创建时间窗口大小为2的数据集
data = []
for i in range(len(result) - 2):
    data.append(result.iloc[i:i + 2].values)

print(len(data))
# 将数据集转换为3D张量
data = np.array(data)

# 分割训练集和测试集
split = int(len(data) * 0.8)
X_train = data[:split, :, :2]
y_train = labels[:split]

X_test = data[split:, :, :2]
y_test = labels[split:]

# print(X_test)
# print(y_test)

# 创建模型
model = Sequential()
model.add(LSTM(128, input_shape=(2, 2)))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model.fit(X_train, y_train, epochs=10, batch_size=32)


# 评估模型
loss, acc = model.evaluate(X_test, y_test, batch_size=32)
print('Test loss:', loss)
print('Test accuracy:', acc)



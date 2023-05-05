import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter
import pickle

df_train = pd.read_csv(r'C:\Users\zhichen.liang19\Desktop\nsl-kdd\KDDTrain+_20Percent.txt')
df_test = pd.read_csv(r'C:\Users\zhichen.liang19\Desktop\nsl-kdd\KDDTest+.txt')
columns = (['duration'
    , 'protocol_type'
    , 'service'
    , 'flag'
    , 'src_bytes'
    , 'dst_bytes'
    , 'land'
    , 'wrong_fragment'
    , 'urgent'
    , 'hot'
    , 'num_failed_logins'
    , 'logged_in'
    , 'num_compromised'
    , 'root_shell'
    , 'su_attempted'
    , 'num_root'
    , 'num_file_creations'
    , 'num_shells'
    , 'num_access_files'
    , 'num_outbound_cmds'
    , 'is_host_login'
    , 'is_guest_login'

    , 'count'
    , 'srv_count'
    , 'serror_rate'
    , 'srv_serror_rate'
    , 'rerror_rate'
    , 'srv_rerror_rate'
    , 'same_srv_rate'
    , 'diff_srv_rate'
    , 'srv_diff_host_rate'

    , 'dst_host_count'
    , 'dst_host_srv_count'
    , 'dst_host_same_srv_rate'
    , 'dst_host_diff_srv_rate'
    , 'dst_host_same_src_port_rate'
    , 'dst_host_srv_diff_host_rate'
    , 'dst_host_serror_rate'
    , 'dst_host_srv_serror_rate'
    , 'dst_host_rerror_rate'
    , 'dst_host_srv_rerror_rate'
    , 'attack'
    , 'level'])
df_train.columns = columns
df_test.columns = columns
dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm','worm']
probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
privilege_attacks = ['buffer_overflow', 'loadmdoule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
access_attacks = ['ftp_write', 'guess_passwd', 'http_tunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xclock', 'xsnoop']
def map_attack(attack):
    if attack in dos_attacks:
        # dos_attacks map to 4
        attack_type = 4
    elif attack in probe_attacks:
        # probe_attacks mapt to 1
        attack_type = 1
    elif attack in privilege_attacks:
        # privilege escalation attacks map to 3
        attack_type = 3
    elif attack in access_attacks:
        # remote access attacks map to 2
        attack_type = 2
    else:
        # normal maps to 0
        attack_type = 0
    return attack_type
attack_map = df_train.attack.apply(map_attack)
df_train['attack_map'] = attack_map
test_attack_map = df_test.attack.apply(map_attack)
df_test['attack_map'] = test_attack_map
df_train = df_train['attack_map'].tolist()
df_test = df_test['attack_map'].tolist()

# n-gram 模型中的 n 值，即连续元素的个数
n = 5
# 将 train 和 test 中的元素序列转换为 n-gram 序列
train_ngrams = [tuple(df_train[i:i+n]) for i in range(len(df_train)-n+1)]
test_ngrams = [tuple(df_test[i:i+n]) for i in range(len(df_test)-n+1)]
# 计算 train 和 test 中每个 n-gram 的出现频率
train_ngram_counter = Counter(train_ngrams)
test_ngram_counter = Counter(test_ngrams)
# 找出 train 中出现频率较低的 n-gram
low_freq_ngrams = [ngram for ngram in train_ngram_counter.keys() if train_ngram_counter[ngram] < test_ngram_counter[ngram]]
# 将 train 中出现频率较低的 n-gram 替换为 test 中出现频率较高的 n-gram
for ngram in low_freq_ngrams:
    most_common_ngram = test_ngram_counter.most_common()[0][0]
    train_indices = [i for i in range(len(train_ngrams)) if train_ngrams[i] == ngram]
    for i in train_indices:
        df_train[i:i+n] = most_common_ngram



# 构造训练数据和测试数据
train_set = set(df_train)
test_set = set(df_test)
common_set = train_set.intersection(test_set)

# 建立元素和索引之间的映射
index_to_element = list(common_set)
element_to_index = {element: index for index, element in enumerate(index_to_element)}

# 将 train 和 test 中的元素转换为对应的索引
train_indices = [element_to_index[element] for element in df_train if element in common_set]
test_indices = [element_to_index[element] for element in df_test if element in common_set]

# 定义模型超参数
input_dim = 1
num_classes = len(common_set)
hidden_size = 64

# 将训练数据转换为模型输入格式
def create_dataset(indices):
    data = np.array(indices)
    data = np.expand_dims(data, axis=1)
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=3,  # 这里设定每个时间窗口的长度为3
        batch_size=32,
        shuffle=False
    )
    dataset = dataset.map(lambda x: (x[:-1], x[-1]))  # 将每个时间窗口的最后一个元素作为目标值
    return dataset

# 构建 lstm 模型
model = keras.Sequential([
    keras.layers.LSTM(hidden_size, input_shape=(3, input_dim)),
    keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')

# 将训练数据转换为模型输入格式
train_dataset = create_dataset(train_indices)
test_dataset = create_dataset(test_indices)

# 训练 lstm 模型
model.fit(train_dataset, epochs=50, verbose=0)

# 在测试数据上计算准确率
y_true = np.array(test_indices[2:])
test_dataset = test_dataset.map(lambda x, y: x)  # 将目标值从数据集中分离出来
y_pred_one_hot = model.predict(test_dataset)
y_pred = np.argmax(y_pred_one_hot, axis=1)
accuracy = np.mean(y_true == y_pred)

print(f'Test accuracy: {accuracy}')

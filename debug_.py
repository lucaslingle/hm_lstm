from HM_LSTM_Cell import HM_LSTM_Cell
from Multi_HM_LSTM_Cell import Multi_HM_LSTM_Cell
import tensorflow as tf
import pandas as pd
import csv
import numpy as np
from collections import Counter

####### HYPERPARAMS #######
batch_size = 64
V = 128
J = 100
emb_dim = 128
hidden_dim = 510
forget_bias = 1.0
learning_rate = 0.002
grad_clip = 5.0
init_slope_value = 1.0
max_slope = 5.0
slope_annealing_increase_per_epoch = 0.04


####### DATASET LOADER #######
PROVIDER_COLUMN_NAMES = ['provider_label', 'provider_comment']
fp = '/Users/lucaslingle/git/estimator_stpeters/datasets/comments_v3.csv'

df = pd.read_csv(fp, sep='\t', names=PROVIDER_COLUMN_NAMES, skiprows=1, quoting=csv.QUOTE_NONE, quotechar='|', escapechar='\\')
df = df[df['provider_label'].apply(lambda x: x in ['INFORMATION_OPERATION', 'ORDINARY_USE'])]

coms = df['provider_comment'].tolist()
counter = Counter()
for comment in coms:
    counter.update(list(comment))
print(counter)
vocab = [k for k, v in counter.items() if v >= 20]
end_of_text = '\x01'
pad = '\x02'
end_of_padded_comment = '\x03'
unk = '\x04'

vocab.append(end_of_text)
vocab.append(pad)
vocab.append(end_of_padded_comment)
vocab.append(unk)
print(vocab)

int2char = {i: c for i, c in enumerate(vocab)}
char2int = {c: i for i, c in enumerate(vocab)}
print(int2char)
print(char2int)

df['good_chars_comment'] = df['provider_comment'].apply(lambda x: ''.join([c for c in x if ord(c) < 128]))
df['standardized_comment'] = df['good_chars_comment'].apply(lambda x: ''.join(list(x) + ['\x01'] + ['\x02' for _ in range(0, max(0, J-len(x)-1))] + ['\x03'])[0:J+1])

df_filtered = df[df['standardized_comment'].apply(lambda x: end_of_padded_comment in x[0:J+1])].copy(deep=True)
df_filtered['comment_ints'] = df_filtered['standardized_comment'].apply(lambda comment: [(char2int[c] if c in char2int else char2int[unk]) for c in comment])

nr_filtered_provider_records = df_filtered.shape[0]
dataset_size = batch_size * (nr_filtered_provider_records // batch_size)
print("foooo")
print(df_filtered)

###### HM-LSTM #######
slope_annealing_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
xs = tf.placeholder(tf.int32, [batch_size, J])
ys = tf.placeholder(tf.int32, [batch_size, J])

'''
layer1 = HM_LSTM_Cell(num_units=hidden_dim, slope_annealing_placeholder=slope_annealing_placeholder, forget_bias=forget_bias)
layer2 = HM_LSTM_Cell(num_units=hidden_dim, slope_annealing_placeholder=slope_annealing_placeholder, forget_bias=forget_bias)
layer3 = HM_LSTM_Cell(num_units=hidden_dim, slope_annealing_placeholder=slope_annealing_placeholder, forget_bias=forget_bias)
multi_cell = Multi_HM_LSTM_Cell([layer1, layer2, layer3])
'''

layer1 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, forget_bias=forget_bias)
layer2 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, forget_bias=forget_bias)
layer3 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, forget_bias=forget_bias)
multi_cell = tf.contrib.rnn.MultiRNNCell([layer1, layer2, layer3])

emb_mat = tf.get_variable('emb_mat', dtype=tf.float32, shape=[V, emb_dim])
emb_xs = tf.nn.embedding_lookup(emb_mat, xs)

initial_state = multi_cell.zero_state(batch_size, tf.float32)
outputs, _ = tf.nn.dynamic_rnn(cell=multi_cell, inputs=emb_xs, initial_state=initial_state)

'''
h_layer1, h_layer2, h_layer3 = outputs

print("each h_layer output should have shape [batch_size, timesteps, hidden dim]")
print(h_layer1.get_shape().as_list())

h_layer1_per_char = tf.reshape(h_layer1, [-1, hidden_dim])
h_layer2_per_char = tf.reshape(h_layer2, [-1, hidden_dim])
h_layer3_per_char = tf.reshape(h_layer3, [-1, hidden_dim])

h_out_per_char = tf.concat([h_layer1_per_char, h_layer2_per_char, h_layer3_per_char], 1)

g1 = tf.layers.dense(h_out_per_char, units=1, activation=tf.nn.sigmoid)
g2 = tf.layers.dense(h_out_per_char, units=1, activation=tf.nn.sigmoid)
g3 = tf.layers.dense(h_out_per_char, units=1, activation=tf.nn.sigmoid)

output_emb = tf.layers.dense(
    tf.concat([
        g1 * h_layer1_per_char, 
        g2 * h_layer2_per_char, 
        g3 * h_layer3_per_char
    ], 1), 
    units=hidden_dim, 
    activation=tf.nn.relu
)
input_to_logits = output_emb
'''
input_to_logits = tf.reshape(outputs, [batch_size * J, -1])

logits = tf.layers.dense(input_to_logits, units=V, activation=None, use_bias=False)
probabilities = tf.nn.softmax(logits, 1)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ys, logits=tf.reshape(logits, [batch_size, J, V]))
loss = tf.reduce_mean(tf.reduce_sum(losses, 1), 0)

#train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
train_op = optimizer.apply_gradients(zip(gradients, variables))

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

nr_epochs = 10

for epoch in range(0, nr_epochs):
    slope_value = min(max_slope, init_slope_value + slope_annealing_increase_per_epoch * epoch)

    for batch_idx in range(0, dataset_size, batch_size):

        comment_ints_batch = df_filtered['comment_ints'].iloc[batch_idx:batch_idx+batch_size].tolist()
        comment_ints_batch = np.array(comment_ints_batch, dtype=np.int32)

        xs_batch = comment_ints_batch[:,0:J]
        ys_batch = comment_ints_batch[:,1:(J+1)]

        feed_dict = {xs: xs_batch, ys: ys_batch, slope_annealing_placeholder: slope_value}

        loss_batch = sess.run(
            loss,
            feed_dict=feed_dict
        )
        print("Epoch {} / {}... batch_idx {}... training loss: {}".format(epoch, nr_epochs, batch_idx, loss_batch))

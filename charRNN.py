
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import random
import string


# In[2]:

f = open('drug_names.txt','r')
input_text = f.read()
chars = list(set(input_text))
word_list = input_text.split('\n')
word_list = list(filter(lambda x : len(x) >= 5, word_list))


# In[3]:

#hyperameters
seq_length = 4
learning_rate = 1e-1
state_size = 8
vocab_size = len(chars)
batch_size = 1
num_epochs = 1000


# In[16]:

def one_hot(character):
    vec = np.zeros((vocab_size,1),dtype=np.float32)
    vec[chars.index(character)] = 1
    return vec


# In[5]:

def word_mat(word):
    mat = np.array([one_hot(char) for char in word]).transpose()
    return mat


# In[ ]:




# In[6]:

X = tf.placeholder(tf.float32, [vocab_size, seq_length])
y = tf.placeholder(tf.float32, [vocab_size, seq_length])

Wxh1 = tf.Variable(tf.truncated_normal([state_size, vocab_size]), name='Wxh1')
Whh1 = tf.Variable(tf.truncated_normal([state_size, state_size]), name ='Whh1')
bh1 = tf.Variable(tf.truncated_normal([state_size, 1]))

Why1 = tf.Variable(tf.truncated_normal([vocab_size, state_size]), name='Why1')
by1 = tf.Variable(tf.truncated_normal([vocab_size, 1]))

initial_state1 = tf.placeholder(tf.float32, [state_size,1])


# In[7]:

x_predict = tf.placeholder(tf.float32, [vocab_size,1])
current_state_predict = tf.placeholder(tf.float32, [state_size, 1])

y_logit = tf.matmul(Why1,current_state_predict) + by1
y_predict = tf.nn.softmax(y_logit,dim=0)
next_state_predict = tf.tanh(tf.matmul(Wxh1, x_predict) + tf.matmul(Whh1, current_state_predict) + bh1)


# In[8]:

input_series = tf.unstack(X, axis=1)
output_series = tf.unstack(y, axis=1)


# In[9]:

current_state = initial_state1
prediction_series = []
logit_series = []

for current_input in input_series:
    current_input = tf.reshape(current_input, [vocab_size, 1])
    
    next_state = tf.tanh(tf.matmul(Wxh1, current_input) + tf.matmul(Whh1, current_state) + bh1)
    logit = tf.matmul(Why1, current_state) + by1
    logit_series.append(logit)
    prediction = tf.nn.softmax(logit,dim=0)
    prediction_series.append(prediction)
    current_state = next_state

losses = [tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=tf.transpose(logit)) for label, logit in zip(output_series, logit_series)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdamOptimizer(1e-2).minimize(total_loss)


# In[10]:

def generate_shifted_pairs(word, seq_length):
    shift_words = []
    for i in range(len(word) - seq_length):
        x = word[i: i + seq_length]
        y = word[i+1 : i + seq_length + 1]
        shift_words.append((x,y))
    return shift_words
generate_shifted_pairs('xanax', 3)
cur_char = 'x'
h = np.zeros((state_size,1), dtype=np.float32)


# In[11]:
#
# def generate(h, char_in, session):
#     x = one_hot(char_in)
#     mm1 = tf.matmul(Wxh, x)
#     mm2 = tf.matmul(Whh, h)
#     h_next = tf.tanh(mm1 + mm2 + bh)
#     pred = tf.nn.softmax(tf.matmul(Why, h) + by)
#     pred_char = chars[tf.argmax(pred).eval(session=session)[0]]
#     return h_next, pred_char


# In[ ]:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(num_epochs):
        _current_state = np.zeros((state_size, 1))
        random.shuffle(word_list)
        long_list = word_list
        random.shuffle(word_list)
        long_list += word_list
        for word in long_list:
            for shifted_pair in generate_shifted_pairs(word, seq_length):
                x_shift = np.reshape(word_mat(shifted_pair[0]), [vocab_size, seq_length])
                y_shift = np.reshape(word_mat(shifted_pair[1]), [vocab_size, seq_length])
                _, _total_loss, _current_state, _pred_series = sess.run([train_step, total_loss, current_state, prediction_series], feed_dict={X:x_shift, y:y_shift, initial_state1:_current_state})
                
        print(_total_loss)
        for j in range(5):
            seed_char = random.choice(string.ascii_lowercase)
            predict_char = one_hot(seed_char)
            seed_state = np.zeros([state_size, 1])
            generated = seed_char
            for i in range(random.randint(5,10)):
                _y, _logit, seed_state = sess.run([y_predict, y_logit, next_state_predict], feed_dict={x_predict:predict_char, current_state_predict:seed_state})
                generated += chars[np.argmax(_y)]
                predict_char = one_hot(chars[np.argmax(_y)])
            print(generated)
            
    
                
        


# In[ ]:




# In[ ]:




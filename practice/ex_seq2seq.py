#%%
# --tokenization--
from tensorflow.keras.preprocessing.text import Tokenizer
paper = ['많은 것을 바꾸고 싶다면 많은 것을 받아들여라.']
tknz = Tokenizer()
tknz.fit_on_texts(paper)
print(tknz.word_index)
print(tknz.word_counts)
# %%
# --word to vector--
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
paper = ['많은 것을 바꾸고 싶다면 많은 것을 받아들여라.']
tknz = Tokenizer()
tknz.fit_on_texts(paper)

idx_paper = tknz.texts_to_sequences(paper)
print(idx_paper)
n = len(tknz.word_index) + 1
print(n)
idx_onehot = to_categorical(idx_paper, num_classes = n)
print(idx_onehot)
#%%
# --word embedding--
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding

model = Sequential()
model.add(Embedding(input_dim=n, output_dim=3))
model.compile(optimizer='rmsprop', loss='mse')
embedding = model.predict(idx_paper)
print(embedding)

# %%
############################
#%%
# --random seed--
import numpy as np
import tensorflow as tf
np.random.seed(0)
tf.random.set_seed(0)
#%%
# --모형 변수 설정--
n_batch = 64
epochs = 100
latent_dim = 256
n_max_sample = 10000
data_path = ("E:\Data\\nlp\\fra.txt")

#%%
# --데이터 불러오기--
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
lines[:10]
# %%
# --data preprocessing (tokenization)--
x_txts = []
y_txts = []
x_chars_uni = set()
y_chars_uni = set()
n_sample = min(n_max_sample, len(lines) - 1)

for line in lines[:n_sample]:
    x_txt, y_txt, _ = line.split('\t')
    y_txt = '\t' + y_txt + '\n'
    x_txts.append(x_txt)
    y_txts.append(y_txt)
    
    for char in x_txt:
        if char not in x_chars_uni:
            x_chars_uni.add(char)
    for char in y_txt:
        if char not in y_chars_uni:
            y_chars_uni.add(char)
            
x_txts[:5]
y_txts[:3]
x_chars_uni
y_chars_uni
# %%
# --토큰 단위 정리--
x_chars_uni = sorted(list(x_chars_uni))
y_chars_uni = sorted(list(y_chars_uni))
n_encoder_tokens = len(x_chars_uni)
n_decoder_tokens = len(y_chars_uni)

max_encoder_seq_len = 0
for txt in x_txts:
    txt_len = len(txt)
    max_encoder_seq_len = max(txt_len, max_encoder_seq_len)
    
max_decoder_seq_len = 0
for txt in y_txts:
    txt_len = len(txt)
    max_decoder_seq_len = max(txt_len, max_decoder_seq_len)
    
print("유니크 인코더 토큰 글자 수: ", n_encoder_tokens)
print("유니크 디코더 토큰 글자 수: ", n_decoder_tokens)
print("인코더 문장 내 최대 문자 수: ", max_encoder_seq_len)
print("디코더 문장 내 최대 문자 수: ", max_decoder_seq_len)

# %%
# --단어 토큰별 인덱스--
x_token_idx = {}
for idx, char in enumerate(x_chars_uni):
    x_token_idx[char] = idx

y_token_idx = {}
for idx, char in enumerate(y_chars_uni):
    y_token_idx[char] = idx
    
x_token_idx
y_token_idx
# %%
# --데이터 영 행렬 만들기--
encoder_x_data = np.zeros((len(x_txts), max_encoder_seq_len, n_encoder_tokens),dtype='float32')

decoder_x_data = np.zeros((len(x_txts), max_decoder_seq_len, n_decoder_tokens), dtype='float32')

decoder_y_data = np.zeros((len(x_txts), max_decoder_seq_len, n_decoder_tokens), dtype='float32')

#%%
# --input data--
for i, x_txt in enumerate(x_txts):
    for t, char in enumerate(x_txt):
        encoder_x_data[i, t, x_token_idx[char]] = 1.
    encoder_x_data[i, t+1:, x_token_idx[' ']] = 1.
    
# %%
# --타깃 데이터 행렬--
for i, y_txt in enumerate(y_txts):
    for t, char in enumerate(y_txt):
        decoder_x_data[i, t, y_token_idx[char]] = 1.
        if t > 0:
            decoder_y_data[i, t-1, y_token_idx[char]] = 1.
    decoder_x_data[i, t+1:, y_token_idx[' ']] = 1.
    decoder_y_data[i, t:, y_token_idx[' ']] = 1.
    
#%%
# --인코더 모형--
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TimeDistributed

encoder_inputs = Input(shape = (None, n_encoder_tokens))
encoder = LSTM(latent_dim, return_state = True)
encoder_outs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

#%%
# --디코더 모형--
decoder_inputs = Input(shape=(None, n_decoder_tokens))
decoder = LSTM(latent_dim, return_sequences=True, return_state = True)
decoder_outs, _, _ = decoder(decoder_inputs, initial_state = encoder_states)
decoder_dense = TimeDistributed(Dense(n_decoder_tokens, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outs)
# %%
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
# %%
# --model compile--
model.compile(optimizer='rmsprop',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

# %%
# --학습--
model.fit([encoder_x_data, decoder_x_data], decoder_y_data,
          batch_size = n_batch,
          epochs=epochs,
          validation_split=0.2)
# %%
# --모델 불러오기--
import os
from tensorflow import keras

os.chdir("E:\Github\\nlp\practice")
model = keras.models.load_model('seq2seq.h5')
model.summary()

#%%
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape = (latent_dim,))
decoder_state_input_c = Input(shape = (latent_dim,))
decoder_states_inputs = [decoder_state_input_h,
                         decoder_state_input_c]
decoder_state_outputs, state_h, state_c = decoder(
    decoder_inputs, initial_state = decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
# %%
# --리버스 인덱스
reverse_x_char_idx = {}
for char, idx in x_token_idx.items():
    reverse_x_char_idx[idx] = char

reverse_y_char_idx = {}
for char, idx in y_token_idx.items():
    reverse_y_char_idx[idx] = char
    
#%%
# --결괏값 디코딩--
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    y_seq = np.zeros((1, 1, n_decoder_tokens))
    y_seq[0, 0, y_token_idx['\t']] = 1.
    
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [y_seq] + states_value)
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_y_char_idx[sampled_token_index]
        decoded_sentence += sampled_char
        
        if (sampled_char == '\n' or
            len(decoded_sentence) > max_decoder_seq_len):
            stop_condition = True

        y_seq = np.zeros((1, 1, n_decoder_tokens))
        y_seq[0, 0, sampled_token_index] = 1.
        
        states_value = [h, c]
    return decoded_sentence

#%%
# --결과 확인--
for seq_idx in range(100):
    x_seq = encoder_x_data[seq_idx : seq_idx + 1]
    decoded_sentence = decode_sequence(x_seq)
    print('-')
    print('Input sentence:', x_txts[seq_idx])
    print('Decoded sentence:', decoded_sentence)
# %%

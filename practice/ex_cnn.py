#%%
import numpy as np
import tensorflow as tf
np.random.seed(0)
tf.random.set_seed(0)

#%%
from tensorflow.keras import datasets
(X_tn0, y_tn0), (X_te0, y_te0) = datasets.mnist.load_data()

# %%
print(X_tn0.shape)
print(y_tn0.shape)
print(X_te0.shape)
print(y_te0.shape)
# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
for i in range(2*5):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_tn0[i].reshape((28, 28)),
               cmap='Greys')
plt.show()
# %%
set(y_tn0)

# %%
# --feature scailing--
X_tn_re = X_tn0.reshape(60000, 28, 28, 1)
X_tn = X_tn_re/255
print(X_tn.shape)
X_te_re = X_te0.reshape(10000, 28, 28, 1)
X_te = X_te_re / 255
print(X_te.shape)
# %%
# --one_hot_encoding--
from tensorflow.keras.utils import to_categorical
y_tn = to_categorical(y_tn0)
y_te = to_categorical(y_te0)

# %%
# --convolution nueral network--
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import MaxPool2D, Flatten
from tensorflow.keras.layers import Dropout

n_class = len(set(y_tn0))

model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5),
                 input_shape = (28, 28, 1),
                 padding='valid',
                 activation='relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, kernel_size = (3, 3),
                 padding = 'valid',
                 activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(n_class, activation = 'softmax'))
model.summary()
# %%
model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

# %%
hist = model.fit(X_tn, y_tn, epochs=3, batch_size = 100)

# %%
print(model.evaluate(X_tn, y_tn)[1])
print(model.evaluate(X_te, y_te)[1])

#%%
y_pred_hot = model.predict(X_te)
print(y_pred_hot[0])

import numpy as np

y_pred = np.argmax(y_pred_hot, axis=1)
print(y_pred_hot)
diff = y_te0 - y_pred
diff_idx = []
y_len = len(y_te0)
for i in range(0, y_len):
    if(diff[i] != 0):
        diff_idx.append(i)
# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
for i in range(2*5):
    plt.subplot(2, 5, i+1)
    raw_idx = diff_idx[i]
    plt.imshow(X_te0[raw_idx].reshape((28, 28)),
               cmap = 'Greys')
plt.show()
# %%

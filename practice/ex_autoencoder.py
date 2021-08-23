#%%
# --set seed--
import numpy as np
import tensorflow as tf
np.random.seed(0)
tf.random.set_seed(0)

# %%
# --data loading--
from tensorflow.keras import datasets
(X_tn0, y_tn0), (X_te0, y_te0) = datasets.mnist.load_data()

#%%
# --EDA--
print(X_tn0.shape)
print(y_tn0.shape)
print(X_te0.shape)
print(y_te0.shape)
# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
for i in range(2*5):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_tn0[i].reshape((28,28)),
               cmap='Greys')
plt.show()
# %%
# --feature scaling--
X_tn_re = X_tn0.reshape(60000,28,28,1)
X_tn = X_tn_re/255
print(X_tn.shape)
X_te_re = X_te0.reshape(10000,28,28,1)
X_te = X_te_re/255
print(X_te.shape)
# %%
# --generating noise feature data--
import numpy as np
X_tn_noise = X_tn + np.random.uniform(-1, 1, size = X_tn.shape)
X_te_noise = X_te + np.random.uniform(-1, 1, size = X_te.shape)

# %%
# --noise data scailing--
X_tn_ns = np.clip(X_tn_noise, a_min=0, a_max=1)
X_te_ns = np.clip(X_te_noise, a_min=0, a_max=1)

#%%
# --noise data visualization--
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
for i in range(2*5):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_tn_ns[i].reshape((28, 28)),
               cmap = 'Greys')
plt.show()
# %%
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Activation

# Encoder
input_layer1 = Input(shape = (28, 28, 1))
x1 = Conv2D(20, kernel_size = (5,5),
            padding='same')(input_layer1)
x1 = Activation(activation='relu')(x1)
output_layer1 = MaxPool2D(pool_size=2,
                          padding = 'same')(x1)
encoder = Model(input_layer1, output_layer1)
encoder.summary()
# %%
print(output_layer1.shape)
print(output_layer1.shape[0])
print(output_layer1.shape[1])
print(output_layer1.shape[2])
print(output_layer1.shape[3])
# %%
# Decoder
input_layer2 = Input(shape = output_layer1.shape[1:4])
x2 = Conv2D(10, kernel_size = (5,5),
            padding='same')(input_layer2)
x2 = Activation(activation = 'relu')(x2)
x2 = UpSampling2D()(x2)
x2 = Conv2D(1, kernel_size = (5,5),
            padding='same')(x2)
output_layer2 = Activation(activation='relu')(x2)
decoder = Model(input_layer2, output_layer2)
decoder.summary()

#%%
# --auto encoder--
input_auto = Input(shape=(28,28,1))
output_auto = decoder(encoder(input_auto))
auto_encoder = Model(input_auto, output_auto)
auto_encoder.summary()

# %%
# --compile--
auto_encoder.compile(loss = 'mean_squared_error',
                     optimizer = 'adam',
                     metrics = ['mean_squared_error'])
# %%
hist = auto_encoder.fit(X_tn_ns, X_tn,
                 epochs=1,
                 batch_size=100)

# %%
X_pred = auto_encoder.predict(X_tn_ns)

#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
for i in range(2*5):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_pred[i].reshape((28,28)),
               cmap='Greys')
plt.show()
# %%

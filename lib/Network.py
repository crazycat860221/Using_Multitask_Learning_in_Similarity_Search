import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LeakyReLU, Lambda

#== First Stage Model
class IndexModel(Model):
  def __init__(self):
    super(IndexModel, self).__init__(self)

    self.embedding = Sequential([
        Dense(256, kernel_initializer='he_uniform', input_dim=32),
        LeakyReLU(alpha=0.3),
        Dropout(0.5), 
        Dense(32),
        Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
    ])
    
    self.indexing = Sequential([
        Dense(512, kernel_initializer='he_uniform'),
        Activation('relu'),
        Dense(512, kernel_initializer='he_uniform'),
        Activation('relu'),
    ])

    self.category = Sequential([
      Dense(4096, kernel_initializer='he_uniform'),
      Activation('softmax')
    ])

    self.distribution = Sequential([
      Dense(1, kernel_initializer='he_uniform'),
      Activation('sigmoid')
    ])

  def call(self, x, mode='indexing'):
    x = self.embedding(x)
    if mode == 'embedding':
        return x

    x = self.indexing(x)

    cate = self.category(x)
    dist = self.distribution(x)
    return cate, dist

#== Second Stage Model
class ResidualModel(Model):
  def __init__(self):
    super(ResidualModel, self).__init__(self)

    self.indexing = Sequential([
        Dense(512, kernel_initializer='he_uniform', input_dim=64),
        Activation('relu'),
        Dense(512, kernel_initializer='he_uniform'),
        Activation('relu'),
    ])

    self.category = Sequential([
      Dense(4096, kernel_initializer='he_uniform'),
      Activation('softmax')
    ])


  def call(self, x):
    pred = self.indexing(x)
    pred = self.category(pred)

    return pred
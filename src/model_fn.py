from tensorflow.keras.layers import (
  Input,Dense,Dropout, Flatten,
  BatchNormalization, Conv2D,
  MaxPooling2D, LeakyReLU, Conv1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, load_model, save_model


def build_model(height, width, classes):
  i = Input(shape=(height, width))

  x = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same')(i)
  # x = LeakyReLU(alpha=0.2)(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)

  x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
  # x = LeakyReLU(alpha=0.2)(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)

  x = Flatten()(x)

  x = Dense(128, activation='relu')(x)
  # x = LeakyReLU(alpha=0.2)(x)
  x = BatchNormalization()(x)

  x = Dropout(0.3)(x)
  x = Dense(classes, activation='softmax')(x)

  model = Model(i, x)
  return model
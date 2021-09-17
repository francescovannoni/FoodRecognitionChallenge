from tensorflow.keras import layers
import tensorflow as tf

# creating our net
FILTER = 16
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
CATEGORIES = 16


def unet_model():
    # INPUT LAYER
    inputs = layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))  # PADDING?? PER ORA NO
    # we hve to work with floating point numbers, divide each pixel by 255
    s = layers.Lambda(lambda x: x / 255)(inputs)  # Normalization

    # CONTRACTION PATH
    # first convolutional layer
    c1 = layers.Conv2D(FILTER, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(s)
    # add dropout(10 percent from c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    # second layer of the first step
    c1 = layers.Conv2D(FILTER, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
    # max pooling
    p1 = layers.MaxPool2D((2, 2))(c1)
    # c1 and p1 we have defined the first layer

    # kernel initializer: you have to start with some weights, these are the starting weights then
    # updated in the training #he normal as to do with normal distribution Centered around zero
    # truncated normal is another one (look at the documentation)

    # we add padding because we want the same input and output image size

    # relu sparsity, many zeros, satures, increase efficiency with regars to tiime and space
    # complexity because of sparsty and avoid the vanishing gradient problem

    # introduce the dead relu problem: network components most luckily neveer updates to a new value

    # UNET SEQUENTIAL OPERATOR

    c2 = layers.Conv2D(FILTER*2, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    c2 = layers.Conv2D(FILTER*2, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
    p2 = layers.MaxPool2D((2, 2))(c2)

    c3 = layers.Conv2D(FILTER*4, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    c3 = layers.Conv2D(FILTER*4, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
    p3 = layers.MaxPool2D((2, 2))(c3)

    c4 = layers.Conv2D(FILTER*8, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    c4 = layers.Conv2D(FILTER*8, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
    p4 = layers.MaxPool2D((2, 2))(c4)

    c5 = layers.Conv2D(FILTER*16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)
    c5 = layers.Conv2D(FILTER*16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)
    # ultimo è a 256

    # starting expansive path
    # ultimo era 256 quindi ridimezza 128
    # deconvolution is the opposite part of convolution
    u6 = layers.Conv2DTranspose(FILTER*8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(FILTER*8, (3, 3), kernel_initializer='he_normal', activation="relu", padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6)
    c6 = layers.Conv2D(FILTER*8, (3, 3), kernel_initializer='he_normal', activation="relu", padding='same')(c6)

    u7 = layers.Conv2DTranspose(FILTER*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(FILTER*4, (3, 3), kernel_initializer='he_normal', activation="relu", padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation('relu')(c7)
    c7 = layers.Conv2D(FILTER*4, (3, 3), kernel_initializer='he_normal', activation="relu", padding='same')(c7)

    u8 = layers.Conv2DTranspose(FILTER*2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(FILTER*2, (3, 3), kernel_initializer='he_normal', activation="relu", padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation('relu')(c8)
    c8 = layers.Conv2D(FILTER*2, (3, 3), kernel_initializer='he_normal', activation="relu", padding='same')(c8)

    u9 = layers.Conv2DTranspose(FILTER, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(FILTER, (3, 3), kernel_initializer='he_normal', activation="relu", padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Activation('relu')(c9)
    c9 = layers.Conv2D(FILTER, (3, 3), kernel_initializer='he_normal', activation="relu", padding='same')(c9)

    # output

    # optimizer a lot of backpropagation to train model sgd, too mean square you can use more. most often adam
    # loss binary crossentropy
    # metric accuracy
    outputs = layers.Conv2D(CATEGORIES, (1, 1), activation="softmax")(c9)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])


    # is summary num parameters is the number of trainable parameters
  #  model.summary()

    return model




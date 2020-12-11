from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow_addons.layers import *

from losses import *


def CompileModel(model, lossFunction, learning_rate=1e-3):
    if lossFunction == Loss.DICE:
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_loss, metrics=[dice_score])
    elif lossFunction == Loss.CROSSENTROPY:
        model.compile(optimizer=Adam(lr=learning_rate), loss=binary_crossentropy, metrics=[dice_score])
    elif lossFunction == Loss.ACTIVECONTOURS:
        model.compile(optimizer=Adam(lr=learning_rate), loss=Active_Contour_Loss, metrics=[dice_score])
    elif lossFunction == Loss.SURFACEnDice:
        model.compile(optimizer=Adam(lr=learning_rate), loss=surface_loss, metrics=[dice_score])
    elif lossFunction == Loss.FOCALLOSS:
        model.compile(optimizer=Adam(lr=learning_rate), loss=FocalLoss, metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTEDCROSSENTROPY:
        model.compile(optimizer=Adam(lr=learning_rate), loss=weighted_bce_loss, metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTED60CROSSENTROPY:
        model.compile(optimizer=Adam(lr=learning_rate), loss=adjusted_weighted_bce_loss(0.6), metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTED70CROSSENTROPY:
        model.compile(optimizer=Adam(lr=learning_rate), loss=adjusted_weighted_bce_loss(0.7), metrics=[dice_score])
    elif lossFunction == Loss.CROSSENTROPY50DICE50:
        model.compile(optimizer=Adam(lr=learning_rate), loss=cross_and_dice_loss(0.5, 0.5), metrics=[dice_score])
    elif lossFunction == Loss.CROSSENTROPY25DICE75:
        model.compile(optimizer=Adam(lr=learning_rate), loss=cross_and_dice_loss(0.25, 0.75), metrics=[dice_score])
    elif lossFunction == Loss.CROSSENTROPY75DICE25:
        model.compile(optimizer=Adam(lr=learning_rate), loss=cross_and_dice_loss(0.75, 0.25), metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTEDCROSSENTROPY50DICE50:
        model.compile(optimizer=Adam(lr=learning_rate), loss=weighted_cross_and_dice_loss(0.5, 0.5),
                      metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTEDCROSSENTROPY25DICE75:
        model.compile(optimizer=Adam(lr=learning_rate), loss=weighted_cross_and_dice_loss(0.25, 0.75),
                      metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTEDCROSSENTROPY75DICE25:
        model.compile(optimizer=Adam(lr=learning_rate), loss=weighted_cross_and_dice_loss(0.75, 0.25),
                      metrics=[dice_score])
    return model


def EncodingLayer(input_layer,
                  kernels=8,
                  kernel_size=3,
                  use_leaky_ReLU=True,
                  leaky_ReLU_alpha=0.1,
                  pool=True):
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(input_layer)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=leaky_ReLU_alpha)(conv) if use_leaky_ReLU else Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=leaky_ReLU_alpha)(conv) if use_leaky_ReLU else Activation('relu')(conv)
    opposite = conv
    if pool:
        # get tensor shape [width and height]
        shape = conv.get_shape()
        height = shape[1]
        width = shape[2]
        conv = AdaptiveAveragePooling2D((height // 2, width // 2))(conv)
    return opposite, conv


def DecodingLayer(layer_input,
                  skipped_input,
                  kernels=8,
                  kernel_size=3,
                  use_leaky_ReLU=True,
                  leaky_ReLU_alpha=0.1):
    conv = Conv2DTranspose(kernels, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_normal')(layer_input)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=leaky_ReLU_alpha)(conv) if use_leaky_ReLU else Activation('relu')(conv)
    concatenated_input = concatenate([conv, skipped_input], axis=3)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(concatenated_input)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=leaky_ReLU_alpha)(conv) if use_leaky_ReLU else Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=leaky_ReLU_alpha)(conv) if use_leaky_ReLU else Activation('relu')(conv)
    return conv


def UNet4_First5x5(pretrained_weights=None,
                   input_size=(320, 320, 1),
                   kernel_size=3,
                   number_of_kernels=32,
                   loss_function=Loss.CROSSENTROPY,
                   learning_rate=1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    opposite0, enc0 = EncodingLayer(inputs, number_of_kernels, 5)
    opposite1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size)
    opposite2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size)
    _, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, pool=False)
    # decoding
    dec2 = DecodingLayer(enc3, opposite2, number_of_kernels * 4, kernel_size)
    dec1 = DecodingLayer(dec2, opposite1, number_of_kernels * 2, kernel_size)
    dec0 = DecodingLayer(dec1, opposite0, number_of_kernels, kernel_size)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    # plot_model(model, to_file='UNet5_First5x5.png', show_shapes=True, show_layer_names=True)
    return model

"""
This module holds all the neural network models for uatu.

To start, their architecture will be mostly hardcoded, but I may generalize it in the futuere.
"""
import tensorflow as tf

def standard_convnet_init_fn(inputs, training= False):

    print inputs.shape
    #TODO add more customization
    initializer = tf.variance_scaling_initializer(scale=2.0)
    # TODO gotta be a better way to do this?
    prob = tf.cond(training, lambda : 0.5, lambda : 1.0) #should i do some fancier tf stuff?
    # NOTE ask waren if i need separate relus
    layers = [ tf.keras.layers.Conv3D(2,input_shape = (64,64,64,1), kernel_size=62, padding='same',
                                  kernel_initializer=initializer),

               tf.keras.layers.BatchNormalization(axis=1,
                                            gamma_initializer=initializer),
               tf.keras.layers.LeakyReLU(alpha=0.01),
               tf.keras.layers.AveragePooling3D(pool_size=(31, 31, 31), strides = 2),
               tf.layers.Conv3D(12, kernel_size=(28, 28, 28), padding='same',
                                  kernel_initializer=initializer),
               tf.keras.layers.BatchNormalization(axis=1,
                                            gamma_initializer=initializer),
               tf.keras.layers.LeakyReLU(alpha=0.01),
               tf.keras.layers.AveragePooling3D(pool_size=(14, 14, 14), strides = 2),

               tf.layers.Conv3D(64, kernel_size=(6, 6, 6), padding='same',
                                  kernel_initializer=initializer),
               tf.keras.keras.layers.BatchNormalization(axis=1,
                                            gamma_initializer=initializer),
               tf.keras.layers.LeakyReLU(alpha=0.01),
               tf.keras.layers.Conv3D(64, kernel_size=(4, 4, 4), padding='same',
                                  kernel_initializer=initializer),
               tf.keras.layers.BatchNormalization(axis=1,
                                            gamma_initializer=initializer),
               tf.keras.layers.LeakyReLU(alpha=0.01),
               tf.keras.layers.Conv3D(128, kernel_size=(3, 3, 3), padding='same',
                                  kernel_initializer=initializer),
               tf.keras.layers.BatchNormalization(axis=1,
                                            gamma_initializer=initializer),
               tf.keras.layers.LeakyReLU(alpha=0.01),
               tf.keras.layers.Conv3D(128, kernel_size=(2, 2, 2), padding='same',
                                  kernel_initializer=initializer),
               tf.keras.layers.BatchNormalization(axis=1,
                                            gamma_initializer=initializer),
               tf.keras.layers.LeakyReLU(alpha=0.01),
               tf.keras.layers.Flatten(),
               tf.keras.layers.Dense(1024, kernel_initializer=initializer),
               tf.keras.layers.Dropout(rate=prob),
               tf.keras.layers.LeakyReLU(alpha=0.01),

               tf.keras.layers.Dense(256, kernel_initializer=initializer),
               tf.keras.layers.Dropout(rate=prob),
               tf.keras.layers.LeakyReLU(alpha=0.01),

               tf.keras.layers.Dense(2, kernel_initializer=initializer),]
    model = tf.keras.Sequential(layers)

    return model(inputs)


def standard_optimizer_init_fn():
    return tf.train.AdamOptimizer(learning_rate=0.005)

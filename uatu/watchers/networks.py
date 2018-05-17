"""
This module holds all the neural network models for uatu.

To start, their architecture will be mostly hardcoded, but I may generalize it in the futuere.
"""
import tensorflow as tf

def standard_convnet_init_fn(inputs, training= False):

    #TODO add more customization
    initializer = tf.variance_scaling_initializer(scale=2.0)
    # TODO gotta be a better way to do this?
    #prob = tf.cond(training, lambda : 0.5, lambda : 1.0) #should i do some fancier tf stuff?
    axis = -1
    # NOTE ask waren if i need separate relus
    conv1_out = tf.layers.conv3d(inputs, 2, kernel_size=62, padding='same',
                                  kernel_initializer=initializer)
    bn1_out = tf.layers.batch_normalization(conv1_out, axis = axis, training=training)
    lr1_out = tf.nn.leaky_relu(bn1_out, alpha=0.01)
    ap1_out = tf.layers.average_pooling3d(lr1_out, pool_size=(31, 31, 31), strides = 2)
    conv2_out = tf.layers.conv3d(ap1_out, 12, kernel_size=(28, 28, 28), padding='same',
                                  kernel_initializer=initializer)
    bn2_out = tf.layers.batch_normalization(conv2_out, axis = axis, training=training)
    lr2_out = tf.nn.leaky_relu(bn2_out, alpha=0.01)

    ap2_out = tf.layers.average_pooling3d(lr2_out, pool_size=(14, 14, 14), strides = 2)

    conv3_out = tf.layers.conv3d(ap2_out, 64, kernel_size=(6, 6, 6), padding='same',
                      kernel_initializer=initializer)
    bn3_out = tf.layers.batch_normalization(conv3_out, axis = axis, training=training)
    lr3_out = tf.nn.leaky_relu(bn3_out, alpha=0.01)
    conv4_out = tf.layers.conv3d(lr3_out, 64, kernel_size=(4, 4, 4), padding='same',
                      kernel_initializer=initializer)
    bn4_out = tf.layers.batch_normalization(conv4_out, axis = axis, training=training)
    lr4_out = tf.nn.leaky_relu(bn4_out, alpha=0.01)
    conv5_out =  tf.layers.conv3d(lr4_out, 128, kernel_size=(3, 3, 3), padding='same',
                      kernel_initializer=initializer)
    bn5_out= tf.layers.batch_normalization(conv5_out, axis = axis, training=training)
    lr5_out = tf.nn.leaky_relu(bn5_out, alpha=0.01)
    conv6_out = tf.layers.conv3d(lr5_out, 128, kernel_size=(2, 2, 2), padding='same',
                      kernel_initializer=initializer)
    bn6_out = tf.layers.batch_normalization(conv6_out, axis = axis, training= training)
    lr6_out = tf.nn.leaky_relu(bn6_out, alpha=0.01)
    flat_out = tf.layers.flatten(lr6_out)
    dense1_out = tf.layers.dense(flat_out, 1024, kernel_initializer=initializer)
    drop1_out = tf.layers.dropout(dense1_out, training=training)
    lr7_out =  tf.nn.leaky_relu(drop1_out, alpha=0.01)

    dense2_out = tf.layers.dense(lr7_out, 256, kernel_initializer=initializer)
    drop2_out = tf.layers.dropout(dense2_out, training=training)
    lr8_out = tf.nn.leaky_relu(drop2_out, alpha=0.01)
    dense3_out = tf.layers.dense(lr8_out, 2, kernel_initializer=initializer)

    return dense3_out


def standard_convnet_init_ob(inputs, training= False):

    #TODO add more customization
    initializer = tf.variance_scaling_initializer(scale=2.0)
    # TODO gotta be a better way to do this?
    prob = tf.cond(training, lambda : 0.5, lambda : 1.0) #should i do some fancier tf stuff?
    # NOTE ask waren if i need separate relus
    layers = [ tf.keras.layers.Conv3D(2,input_shape = (64,64,64,1), kernel_size=62, padding='same',
                                  kernel_initializer=initializer, name = 'Dickhead'),
               tf.keras.layers.BatchNormalization(name = 'Butts'),
               tf.keras.layers.LeakyReLU(alpha=0.01),
               tf.keras.layers.AveragePooling3D(pool_size=(31, 31, 31), strides = 2),
               tf.keras.layers.Flatten(),
               tf.keras.layers.Dense(2, kernel_initializer=initializer),]
    '''
               tf.layers.Conv3D(12, kernel_size=(28, 28, 28), padding='same',
                                  kernel_initializer=initializer),
               #tf.keras.layers.BatchNormalization(),
               tf.keras.layers.LeakyReLU(alpha=0.01),

               tf.layers.AveragePooling3D(pool_size=(14, 14, 14), strides = 2),

               tf.layers.Conv3D(64, kernel_size=(6, 6, 6), padding='same',
                                  kernel_initializer=initializer),
               #tf.keras.layers.BatchNormalization(),
               tf.keras.layers.LeakyReLU(alpha=0.01),
               tf.layers.Conv3D(64, kernel_size=(4, 4, 4), padding='same',
                                  kernel_initializer=initializer),
               #tf.keras.layers.BatchNormalization(),
               tf.keras.layers.LeakyReLU(alpha=0.01),
               tf.layers.Conv3D(128, kernel_size=(3, 3, 3), padding='same',
                                  kernel_initializer=initializer),
               #tf.keras.layers.BatchNormalization(),
               tf.keras.layers.LeakyReLU(alpha=0.01),
               tf.layers.Conv3D(128, kernel_size=(2, 2, 2), padding='same',
                                  kernel_initializer=initializer),
               #tf.keras.layers.BatchNormalization(),
               tf.keras.layers.LeakyReLU(alpha=0.01),
               tf.layers.Flatten(),
               tf.layers.Dense(1024, kernel_initializer=initializer),
               tf.keras.layers.Dropout(rate=prob),
               tf.keras.layers.LeakyReLU(alpha=0.01),

               tf.layers.Dense(256, kernel_initializer=initializer),
               tf.keras.layers.Dropout(rate=prob),
               tf.keras.layers.LeakyReLU(alpha=0.01),

               tf.layers.Dense(2, kernel_initializer=initializer),]
    '''
    model = tf.keras.Sequential(layers)

    return model(inputs)

def standard_optimizer_init_fn():
    return tf.train.AdamOptimizer(learning_rate=0.005)

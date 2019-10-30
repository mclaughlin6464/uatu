"""
This module holds all the neural network models for uatu.

To start, their architecture will be mostly hardcoded, but I may generalize it in the futuere.
"""

try:
    import tensorflow as tf
    from ConcreteDropout import ConcreteDropout
except:
    print 'failed'
    ConcreteDropout = lambda x: x
    

def standard_convnet_init_fn(inputs, training= False):

    #TODO add more customization
    initializer = tf.variance_scaling_initializer(scale=2.0)
    # TODO gotta be a better way to do this?
    #prob = tf.cond(training, lambda : 0.5, lambda : 1.0) #should i do some fancier tf stuff?
    axis = -1
    # NOTE ask waren if i need separate relus
    conv1_out = tf.layers.conv2d(inputs, 16, kernel_size=7, padding='same',
                                  kernel_initializer=initializer)
    bn1_out = tf.layers.batch_normalization(conv1_out, axis = axis, training=training)
    lr1_out = tf.nn.leaky_relu(bn1_out, alpha=0.01)
    ap1_out = tf.layers.average_pooling2d(lr1_out, pool_size=31, strides = 2)
    conv2_out = tf.layers.conv2d(ap1_out, 32, kernel_size=5, padding='same',
                                  kernel_initializer=initializer)
    bn2_out = tf.layers.batch_normalization(conv2_out, axis = axis, training=training)
    lr2_out = tf.nn.leaky_relu(bn2_out, alpha=0.01)

    ap2_out = tf.layers.average_pooling2d(lr2_out, pool_size=14, strides = 2)

    conv3_out = tf.layers.conv2d(ap2_out, 64, kernel_size=5, padding='same',
                      kernel_initializer=initializer)
    bn3_out = tf.layers.batch_normalization(conv3_out, axis = axis, training=training)
    lr3_out = tf.nn.leaky_relu(bn3_out, alpha=0.01)
    conv4_out = tf.layers.conv2d(lr3_out, 128, kernel_size=3, padding='same',
                      kernel_initializer=initializer)
    bn4_out = tf.layers.batch_normalization(conv4_out, axis = axis, training=training)
    lr4_out = tf.nn.leaky_relu(bn4_out, alpha=0.01)
    conv5_out =  tf.layers.conv2d(lr4_out, 256, kernel_size=3, padding='same',
                      kernel_initializer=initializer)
    bn6_out = tf.layers.batch_normalization(conv5_out, axis = axis, training= training)
    lr6_out = tf.nn.leaky_relu(bn6_out, alpha=0.01)
    ap6_out = tf.layers.average_pooling2d(lr6_out, pool_size=3, strides = 2)
    flat_out = tf.layers.flatten(ap6_out)
    dense1_out = tf.layers.dense(flat_out, 512)# kernel_initializer=initializer)
    drop1_out = tf.layers.dropout(dense1_out, training=training)
    lr7_out =  tf.nn.leaky_relu(dense1_out, alpha=0.01)

    dense2_out = tf.layers.dense(lr7_out, 256, kernel_initializer=initializer)
    drop2_out = tf.layers.dropout(dense2_out, training=training)
    lr8_out = tf.nn.leaky_relu(drop2_out, alpha=0.01)
    dense3_out = tf.layers.dense(lr8_out, 2, kernel_initializer=initializer)

    return dense3_out

def shallow_convnet_init_fn(inputs, training=False):
    # TODO add more customization
    initializer = tf.variance_scaling_initializer(scale=2.0)
    # TODO gotta be a better way to do this?
    # prob = tf.cond(training, lambda : 0.5, lambda : 1.0) #should i do some fancier tf stuff?

    axis = -1

    # NOTE ask waren if i need separate relus
    conv1_out = tf.layers.conv2d(inputs, 16, kernel_size=7, padding='same',
                                 kernel_initializer=initializer)
    bn1_out = tf.layers.batch_normalization(conv1_out, axis=axis, training=training)
    lr1_out = tf.nn.leaky_relu(bn1_out, alpha=0.01)
    ap1_out = tf.layers.average_pooling2d(lr1_out, pool_size=12, strides=2)
    conv2_out = tf.layers.conv2d(ap1_out, 32, kernel_size=5, padding='same',
                                 kernel_initializer=initializer)
    bn2_out = tf.layers.batch_normalization(conv2_out, axis=axis, training=training)
    lr2_out = tf.nn.leaky_relu(bn2_out, alpha=0.01)

    ap2_out = tf.layers.average_pooling2d(lr2_out, pool_size=8, strides=2)

    conv3_out = tf.layers.conv2d(ap2_out, 64, kernel_size=5, padding='same',
                                 kernel_initializer=initializer)
    bn3_out = tf.layers.batch_normalization(conv3_out, axis=axis, training=training)
    lr3_out = tf.nn.leaky_relu(bn3_out, alpha=0.01)
    # conv4_out = tf.layers.conv3d(lr3_out, 64, kernel_size=(4, 4, 4), padding='same')
    #                  kernel_initializer=initializer)
    # bn4_out = tf.layers.batch_normalization(conv4_out, axis = axis, training=training)
    # lr4_out = tf.nn.leaky_relu(bn4_out, alpha=0.01)
    # conv5_out =  tf.layers.conv3d(lr4_out, 128, kernel_size=(3, 3, 3), padding='same')
    #                  kernel_initializer=initializer)
    # bn5_out= tf.layers.batch_normalization(conv5_out, axis = axis, training=training)
    # lr5_out = tf.nn.leaky_relu(bn5_out, alpha=0.01)
    # conv6_out = tf.layers.conv3d(lr5_out, 128, kernel_size=(2, 2, 2), padding='same')
    #                  kernel_initializer=initializer)
    # bn6_out = tf.layers.batch_normalization(conv6_out, axis = axis, training= training)
    # lr6_out = tf.nn.leaky_relu(bn6_out, alpha=0.01)
    flat_out = tf.layers.flatten(lr3_out)
    #dense1_out = tf.layers.dense(flat_out, 1024)  # kernel_initializer=initializer)
    #drop1_out = tf.layers.dropout(dense1_out, training=training)
    #lr7_out = tf.nn.leaky_relu(drop1_out, alpha=0.01)

    dense2_out = tf.layers.dense(flat_out, 256)  # kernel_initializer=initializer)
    #drop2_out = tf.layers.dropout(dense2_out, training=training)
    lr8_out = tf.nn.leaky_relu(dense2_out, alpha=0.01)
    dense3_out = tf.layers.dense(lr8_out, 2)  # kernel_initializer=initializer)

    return dense3_out


class DummyWrapper(object):

    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, training=False):
        return self.model(inputs)

def gupta_network_init_fn(inputs, **kwargs):
    '''
    Emulate the architecture in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.97.103515 and 
    https://arxiv.org/pdf/1806.05995.pdf. 
    '''
    print 'A'
    return gupta_bayesian_network_init_fn(inputs, wrapper = DummyWrapper, nout = 2, **kwargs)

def _conv_relu(input, filters, kernel_size, kernel_initializer, kernel_regularizer):
    """
    Copying this from here https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.7/tensorflow/g3doc/how_tos/variable_scope/index.md
    """
    weights = tf.get_variable("weights", [kernel_size, kernel_size, filters[0], filters[1]],\
                              initializer=kernel_initializer, regularizer=kernel_regularizer)
    #biases = tf.get_variable("biases", [filters[1]])
    conv = tf.nn.conv2d(input, weights, strides = [1,1,1,1], padding = 'SAME') 
    return tf.nn.leaky_relu(conv, alpha=0.01)

def _dense_relu(input, filters, kernel_initializer, kernel_regularizer):
    """
    Copying this from here https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.7/tensorflow/g3doc/how_tos/variable_scope/index.md
    """
    weights = tf.get_variable("weights", [filters[0], filters[1]],\
                              initializer=kernel_initializer, regularizer=kernel_regularizer)
    biases = tf.get_variable("biases", [filters[1]])
    dense = tf.matmul(input, weights) + biases
    return tf.nn.leaky_relu(dense, alpha=0.01)


def gupta_adv_network_init_fn(inputs, training=False, lam=1e-6, wrapper=ConcreteDropout, nout = 2):
    '''
    Emulate the architecture in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.97.103515 and 
    https://arxiv.org/pdf/1806.05995.pdf.
    Implementation of Gal and Garmani approximation
    '''

    initializer = tf.variance_scaling_initializer(scale=2.0)
    regularizer = tf.contrib.layers.l2_regularizer(scale = lam)

    with tf.variable_scope("conv1"):
        lr1_out = _conv_relu(inputs, [1, 32], 3, initializer, regularizer)
    # can't do weight sharing with concrete dropout... cross that bridge later
    #ap1_out = tf.layers.average_pooling2d(lr1_out, pool_size=3, strides=3)
    with tf.variable_scope("conv2"):
        lr2_out = _conv_relu(lr1_out, [32, 64], 3, initializer, regularizer)

    ap2_out = tf.layers.average_pooling2d(lr2_out, pool_size=2, strides=2)

    with tf.variable_scope("conv3"):
        lr3_out = _conv_relu(ap2_out, [64, 128], 3, initializer, regularizer)

    with tf.variable_scope("conv4"):
        lr4_out = _conv_relu(lr3_out, [128, 128], 3, initializer, regularizer)

    ap4_out = tf.layers.average_pooling2d(lr4_out, pool_size=2, strides=2)

    with tf.variable_scope("conv5"):
        lr5_out = _conv_relu(ap4_out, [128, 128], 3, initializer, regularizer)

    with tf.variable_scope("conv6"):
        lr6_out = _conv_relu(lr5_out, [128, 128], 3, initializer, regularizer)

    ap6_out = tf.layers.average_pooling2d(lr6_out, pool_size=2, strides=2)

    flat_out = tf.layers.flatten(ap6_out)
    # TODO I've removed the dropout for the standard network here implicitly

    with tf.variable_scope("dense1"):
        dense1_out = _dense_relu(flat_out, [32*32*128, 256], initializer, regularizer)

    with tf.variable_scope("dense2"):
        dense2_out = _dense_relu(dense1_out, [256, 256], initializer, regularizer)

    with tf.variable_scope("dense3"):
        dense3_out = _dense_relu(dense2_out, [256, nout], initializer, regularizer)

    return dense3_out

def gupta_adv_network_large_filters_init_fn(inputs, training=False, lam=1e-6, wrapper=ConcreteDropout, nout = 2):
    '''
    Emulate the architecture in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.97.103515 and 
    https://arxiv.org/pdf/1806.05995.pdf.
    Implementation of Gal and Garmani approximation
    '''

    initializer = tf.variance_scaling_initializer(scale=2.0)
    regularizer = tf.contrib.layers.l2_regularizer(scale = lam)

    with tf.variable_scope("conv1"):
        lr1_out = _conv_relu(inputs, [1, 32], 64, initializer, regularizer)
    # can't do weight sharing with concrete dropout... cross that bridge later
    #ap1_out = tf.layers.average_pooling2d(lr1_out, pool_size=3, strides=3)
    with tf.variable_scope("conv2"):
        lr2_out = _conv_relu(lr1_out, [32, 64], 32, initializer, regularizer)

    ap2_out = tf.layers.average_pooling2d(lr2_out, pool_size=2, strides=2)

    with tf.variable_scope("conv3"):
        lr3_out = _conv_relu(ap2_out, [64, 128], 16, initializer, regularizer)

    with tf.variable_scope("conv4"):
        lr4_out = _conv_relu(lr3_out, [128, 128], 3, initializer, regularizer)

    ap4_out = tf.layers.average_pooling2d(lr4_out, pool_size=2, strides=2)

    with tf.variable_scope("conv5"):
        lr5_out = _conv_relu(ap4_out, [128, 128], 3, initializer, regularizer)

    with tf.variable_scope("conv6"):
        lr6_out = _conv_relu(lr5_out, [128, 128], 3, initializer, regularizer)

    ap6_out = tf.layers.average_pooling2d(lr6_out, pool_size=2, strides=2)

    flat_out = tf.layers.flatten(ap6_out)
    # TODO I've removed the dropout for the standard network here implicitly

    with tf.variable_scope("dense1"):
        dense1_out = _dense_relu(flat_out, [32*32*128, 256], initializer, regularizer)

    with tf.variable_scope("dense2"):
        dense2_out = _dense_relu(dense1_out, [256, 256], initializer, regularizer)

    with tf.variable_scope("dense3"):
        dense3_out = _dense_relu(dense2_out, [256, nout], initializer, regularizer)

    return dense3_out


def gupta_bayesian_network_init_fn(inputs, training=False, lam=1e-6, wrapper=ConcreteDropout, nout = 5):
    '''
    Emulate the architecture in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.97.103515 and 
    https://arxiv.org/pdf/1806.05995.pdf.
    Implementation of Gal and Garmani approximation
    '''
    print 'B'
    # TODO add more customization
    initializer = tf.variance_scaling_initializer(scale=2.0)
    regularizer = tf.contrib.layers.l2_regularizer(scale = lam)
    axis = -1
    conv1_out = wrapper(tf.layers.Conv2D(32, kernel_size=3, padding='same',\
                            kernel_initializer=initializer, kernel_regularizer=regularizer))(inputs, training=training)
    lr1_out = tf.nn.leaky_relu(conv1_out, alpha=0.01)
    #ap1_out = tf.layers.average_pooling2d(lr1_out, pool_size=3, strides=3)
    conv2_out = wrapper(tf.layers.Conv2D(64, kernel_size=3, padding='same',\
                            kernel_initializer=initializer, kernel_regularizer=regularizer))(lr1_out, training=training)
    lr2_out = tf.nn.leaky_relu(conv2_out, alpha=0.01)
    ap2_out = tf.layers.average_pooling2d(lr2_out, pool_size=2, strides=2)
    conv3_out = wrapper(tf.layers.Conv2D(128, kernel_size=3, padding='same',\
                            kernel_initializer=initializer, kernel_regularizer=regularizer))(ap2_out, training=training)
    lr3_out = tf.nn.leaky_relu(conv3_out, alpha=0.01)
    conv4_out = wrapper(tf.layers.Conv2D(128, kernel_size=3, padding='same',\
                            kernel_initializer=initializer, kernel_regularizer=regularizer))(lr3_out, training=training)
    lr4_out = tf.nn.leaky_relu(conv4_out, alpha=0.01)
    ap4_out = tf.layers.average_pooling2d(lr4_out, pool_size=2, strides=2)
    conv5_out = wrapper(tf.layers.Conv2D(128, kernel_size=3, padding='same',\
                            kernel_initializer=initializer, kernel_regularizer=regularizer))(ap4_out, training=training)
    lr5_out = tf.nn.leaky_relu(conv5_out, alpha=0.01)
    conv6_out = wrapper(tf.layers.Conv2D(128, kernel_size=3, padding='same',\
                            kernel_initializer=initializer, kernel_regularizer=regularizer))(lr5_out, training=training)
    lr6_out = tf.nn.leaky_relu(conv6_out, alpha=0.01)
    ap6_out = tf.layers.average_pooling2d(lr6_out, pool_size=2, strides=2)

    flat_out = tf.layers.flatten(ap6_out)
    # TODO I've removed the dropout for the standard network here implicitly
    dense1_out = wrapper(tf.layers.Dense(256, kernel_initializer=initializer))(flat_out, training=training)
    lr7_out = tf.nn.leaky_relu(dense1_out, alpha=0.01)
    dense2_out = wrapper(tf.layers.Dense(256, kernel_initializer=initializer))(lr7_out, training=training)
    lr8_out = tf.nn.leaky_relu(dense2_out, alpha=0.01)

    dense3_out = wrapper(tf.layers.Dense(nout, kernel_initializer=initializer))(lr8_out, training=training)
    return dense3_out

def standard_optimizer_init_fn(lr = 0.0005):
    return tf.train.AdamOptimizer(learning_rate=lr)

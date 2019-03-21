"""
Test trained networks
"""
try:
    import tensorflow as tf
except:
    pass
import numpy as np
import h5py

def test(model_init_fn, data,n_samples, fname, samples_fname):
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, 256,256,1])

    training = tf.placeholder(tf.bool, name='training')

    preds = model_init_fn(x, training=training)

    with tf.device('/cpu:0'):

        saver = tf.train.Saver()


    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            saver.restore(sess, fname)
        print 'Starting sampling'
        f = h5py.File(samples_fname, 'w')
        f.close()

        for i, (x_np,  y_np) in enumerate(data):
            print i,
            assert y_np.shape[0] == 1 , 'batchsize greater than 1'

            samples = []
            feed_dict = {x: x_np, training: False}

            for j in xrange(n_samples):
                preds_np  = sess.run(preds, feed_dict=feed_dict)
                samples.append(preds_np)

            samples = np.vstack(samples)
            key = key_func(y_np)
            print key
            f = h5py.File(samples_fname)
            if key in f.keys():
                grp = f[key]
            else:
                grp = f.create_group(key)
            n_prev = len(grp.keys())

            grp.create_dataset('Map_%03d'%(n_prev), data = samples)
            f.close()



def key_func(y):
    """
    return a key based on a given y
    :param y: (2,1) numpy array ofr om and s8
    :return:  key, the key corresponding to this y
    """

    return "Om_{om:.6f}_s8_{s8:.6f}".format(om=y[0,0], s8=y[0,1])

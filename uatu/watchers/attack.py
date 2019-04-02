"""
Perform adversarial attacks on images
"""
try:
    import tensorflow as tf
except:
    pass
import numpy as np
import h5py

from .test import key_func

def compute_attacked_maps(model_init_fn, cost_fn, fname, data, target_y_np, target_fname, attrs):

    try:
        f = h5py.File('target_fname', 'w')
        for key in attrs:
            f.attrs[key] = attrs[key]
        f.close()
    except IOError:
        raise IOError("Problem encountered opening %s"%target_fname)

    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 256, 256,1])
    y = tf.placeholder(tf.float32, [None,2])

    training = tf.placeholder(tf.bool, name='training')
    preds = model_init_fn(x, training)

    loss = cost_fn(y, preds)
    grads = tf.gradients(loss, x)

    # learning rate maybe needed?
    dX = tf.divide(grads, tf.norm(grads))

    with tf.device('/cpu:0'):
        saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, fname)

        for i, (x_np,  y_np) in enumerate(data):
            x_attacked_np = x_np.copy()
            for i in xrange(100):
                feed_dict = {x: x_attacked_np,y:target_y_np, training: False}
                #loss_np, update_ops_np = sess.run([loss,update_ops], feed_dict=feed_dict)
                dX_np = sess.run([dX], feed_dict=feed_dict)
                x_attacked_np+=dX_np

            for am, _y_np in zip(x_attacked_np, y_np):
                key = key_func(_y_np)

                f = h5py.File(target_fname)
                if key in f.keys():
                    grp = f[key]
                else:
                    grp = f.create_group(key)

                if 'X' not in grp.keys():
                    x_dset = grp.create_dataset('X', maxshape= (None, x_np.shape[1], x_np.shape[2], x_np.shape[2]))
                    y_dset = grp.create_dataset('Y', maxshape = (None, 2))

                else:
                    x_dset, y_dset = grp['X'], grp['Y']

                l = len(x_dset)

                x_dset.resize((l+1), axis = 0)
                x[-1] = am

                y_dset.resize((l+1), axis = 0)
                y[-1] = _y_np

                f.close()

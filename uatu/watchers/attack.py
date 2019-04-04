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

def compute_attacked_maps(model_init_fn, cost_fn, network_fname, data, target_y_np, target_fname, attrs):

    try:
        f = h5py.File(target_fname, 'w')
        for key in attrs:
            f.attrs[key] = attrs[key]
        f.close()
    except IOError:
        raise IOError("Problem encountered opening %s"%target_fname)

    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 256, 256,1])
    y = tf.placeholder(tf.float32, [None,2])

    training = tf.placeholder(tf.bool, name='training')
    preds = model_init_fn(x, training=False)

    loss = cost_fn(y, preds)
    grads = tf.gradients(loss, x)

    # learning rate maybe needed?
    dX = tf.divide(grads, tf.norm(grads))

    with tf.device('/cpu:0'):
        saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, network_fname)
        key_dict = {}
        for i, (x_np,  y_np) in enumerate(data):
            x_attacked_np = x_np.copy()
            for i in xrange(100):
                feed_dict = {x: x_attacked_np,y:target_y_np, training: False}
                #loss_np, update_ops_np = sess.run([loss,update_ops], feed_dict=feed_dict)
                dX_np = sess.run([dX], feed_dict=feed_dict)[0][0]
                x_attacked_np+=dX_np

            for am, _y_np in zip(x_attacked_np, y_np):
                key = key_func(_y_np.reshape((1, 2)))
                if key not in key_dict:
                    key_dict[key] = len(key_dict)

                box_key = 'Box%03d'%key_dict[key]

                f = h5py.File(target_fname)
                if box_key in f.keys():
                    grp = f[box_key]
                else:
                    grp = f.create_group(box_key)

                if 'X' not in grp.keys():
                    x_dset = grp.create_dataset('X', data = am.reshape((1, am.shape[0], am.shape[1], am.shape[2])), maxshape= (None, x_np.shape[1], x_np.shape[2], x_np.shape[3]))
                    y_dset = grp.create_dataset('Y', data = _y_np.reshape((1,2)), maxshape = (None, 2))

                else:
                    x_dset, y_dset = grp['X'], grp['Y']

                    l = len(x_dset)

                    x_dset.resize((l+1), axis = 0)
                    x_dset[-1] = am

                    y_dset.resize((l+1), axis = 0)
                    y_dset[-1] = _y_np

                f.close()

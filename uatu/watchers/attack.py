"""
Perform adversarial attacks on images
"""
try:
    import tensorflow as tf
except:
    pass
import numpy as np
import h5py
from scipy.ndimage import rotate

from .test import key_func

def log_barrier(x_p, x_o, eps, lam):

    norm = tf.norm(x_p - x_o+1e-6)
    return -tf.log(eps - norm )/lam

def compute_attacked_maps(model_init_fn, cost_fn, network_fname, data, target_y_np,\
                          target_fname, attrs, use_log_barrier = True, log_eps = 1.5):

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

    x_orig = tf.placeholder(tf.float32, [None, 256, 256,1])
    log_barrier_weight = tf.placeholder(tf.float32)

    #training = tf.placeholder(tf.bool, name='training')
# TODO may have to put scope in here now 
    preds = model_init_fn(x, training=False)

    loss = cost_fn(y, preds)

    if use_log_barrier:
        barrier_loss = log_barrier(x, x_orig, log_eps, log_barrier_weight)
        loss = loss + barrier_loss

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
            #x_orig_power = x_attacked_np.mean()

            step = 1e-3
            lam = 1e9
            lam_incr = 1.00001

            for i in range(100):
                feed_dict = {x: x_attacked_np,y:target_y_np, x_orig: x_np, log_barrier_weight: lam }
                #loss_np, update_ops_np = sess.run([loss,update_ops], feed_dict=feed_dict)
                dX_np, loss_np = sess.run([dX, loss], feed_dict=feed_dict)#[0][0]
                # initially had a + that seemed wrong
                # TODO adding transpose to try to test out if rotations alter training
                x_attacked_np-=step*dX_np[0]

                lam*=lam_incr
                if lam >= 1e9:
                    lam = 1e9
            
            # ensure the attacked map has the same normalization as the old one.
            #x_attacked_np = x_attacked_np*x_orig_power/x_attacked_np.mean()

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


def compute_shuffled_attacked_maps(model_init_fn, cost_fn, network_fname, data,\
                                   true_to_target_map, target_fname, attrs,\
                                   use_log_barrier=True, log_eps = 1.5):
    try:
        f = h5py.File(target_fname, 'w')
        for key in attrs:
            f.attrs[key] = attrs[key]
        f.close()
    except IOError:
        raise IOError("Problem encountered opening %s" % target_fname)

    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 256, 256, 1])
    y = tf.placeholder(tf.float32, [None, 2])

    x_orig = tf.placeholder(tf.float32, [None, 256, 256, 1])
    # training = tf.placeholder(tf.bool, name='training')
    with tf.variable_scope('kappa_filters') as scope:
        preds = model_init_fn(x, training=False)

    loss = cost_fn(y, preds)
    log_barrier_weight = tf.placeholder(tf.float32)

    if use_log_barrier:
        barrier_loss = log_barrier(x, x_orig, log_eps, log_barrier_weight)
        loss = loss + barrier_loss
    grads = tf.gradients(loss, x)

    # learning rate maybe needed?
    dX = tf.divide(grads, tf.norm(grads))

    with tf.device('/cpu:0'):
        saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, network_fname)
        key_dict = {}

        step = 1e-3
        lam = 1e9
        lam_incr = 1.00001

        for i, (x_np, y_np) in enumerate(data):
            assert y_np.shape[0] == 1, "Doesn't support batching"
            x_attacked_np = x_np.copy()
            x_orig_power = x_attacked_np.mean()
            target_y_np = true_to_target_map[tuple(y_np.squeeze())].reshape((1,-1))
            for i in range(100):
                feed_dict = {x: x_attacked_np, y: target_y_np, x_orig: x_np, log_barrier_weight: lam }  # , training: False}
                # loss_np, update_ops_np = sess.run([loss,update_ops], feed_dict=feed_dict)
                dX_np, loss_np = sess.run([dX, loss], feed_dict=feed_dict)  # [0][0]
                # initially had a + that seemed wrong
                # TODO add the log barrier to these
                x_attacked_np -= step*dX_np[0]#rotate(dX_np[0], 0, (2, 1))

                lam *= lam_incr
                if lam >= 1e9:
                    lam = 1e9
            # ensure the attacked map has the same normalization as the old one.
            #x_attacked_np = x_attacked_np * x_orig_power / x_attacked_np.mean()

            # simplify rename
            am = x_attacked_np

            key = key_func(y_np.reshape((1, 2)))
            if key not in key_dict:
                key_dict[key] = len(key_dict)

            box_key = 'Box%03d' % key_dict[key]

            f = h5py.File(target_fname)
            if box_key in f.keys():
                grp = f[box_key]
            else:
                grp = f.create_group(box_key)

            if 'X' not in grp.keys():
                x_dset = grp.create_dataset('X', data=am.reshape((1, am.shape[1], am.shape[2], am.shape[3])),
                                            maxshape=(None, am.shape[1], am.shape[2], am.shape[3]))
                y_dset = grp.create_dataset('true_Y', data=y_np.reshape((1, 2)), maxshape=(None, 2))
                ty_dset = grp.create_dataset('target_Y', data=target_y_np.reshape((1, 2)), maxshape=(None, 2))


            else:
                x_dset, y_dset, ty_dset = grp['X'], grp['true_Y'], grp['target_Y']

                l = len(x_dset)

                x_dset.resize((l + 1), axis=0)
                x_dset[-1] = am

                y_dset.resize((l + 1), axis=0)
                y_dset[-1] = y_np

                ty_dset.resize((l + 1), axis=0)
                ty_dset[-1] = target_y_np

            f.close()

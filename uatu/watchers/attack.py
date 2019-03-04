"""
Perform adversarial attacks on images
"""
try:
    import tensorflow as tf
except:
    pass
import numpy as np

def compute_attacked_maps(model_init_fn, cost_fn, fname, data, target_y_np):

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
        attacked_maps = []
        for i, (x_np,  _) in enumerate(data):
            x_attacked_np = x_np.copy()
            for i in xrange(100):
                feed_dict = {x: x_attacked_np,y:target_y_np, training: False}
                #loss_np, update_ops_np = sess.run([loss,update_ops], feed_dict=feed_dict)
                dX_np = sess.run([dX], feed_dict=feed_dict)
                x_attacked_np+=dX_np

            attacked_maps.append(x_attacked_np)

        return np.array(attacked_maps)

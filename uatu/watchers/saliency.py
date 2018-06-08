"""
Make saliency maps
"""
try:
    import tensorflow as tf
except:
    pass
import numpy as np

def compute_saliency_maps(model_init_fn, cost_fn, device, fname, data):

    tf.reset_default_graph()
    with tf.device(device):
        x = tf.placeholder(tf.float32, [None, 64,64,64,1])
        y = tf.placeholder(tf.float32, [None,2])

        training = tf.placeholder(tf.bool, name='training')

        preds = model_init_fn(x, training)

        #correct_scores = tf.gather_nd(model.scores,
        #                              tf.stack((tf.range(X.shape[0]), model.labels), axis=1))

        loss = cost_fn(y, preds)
        grads = tf.gradients(loss, x)

    with tf.device('/cpu:0'):
        saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, fname)
        saliencies = []
        for i, (x_np,  y_np) in enumerate(data):
            feed_dict = {x: x_np,y:y_np, training: False}
            #loss_np, update_ops_np = sess.run([loss,update_ops], feed_dict=feed_dict)
            grads_np = sess.run([grads], feed_dict=feed_dict)[0][0]
            saliency = np.max(np.abs(grads_np), axis=-1)

            saliencies.append(saliency)

        return np.array(saliencies)

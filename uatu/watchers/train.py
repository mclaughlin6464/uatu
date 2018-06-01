"""
Train the neural network passed in.
Mostly copied from the CS231n notes
"""

try:
    import tensorflow as tf
except:
    pass
import numpy as np

def standard_cost_fn(y, preds):
    return tf.losses.mean_squared_error(labels=y, predictions=preds, reduction=tf.losses.Reduction.SUM)

def bayes_cost_fn(y, preds):
    log_s1 = tf.slice(preds, [0, 2], [-1, 1])
    log_s2 = tf.slice(preds, [0, 3], [-1, 1])
    rho = tf.tanh(tf.slice(preds, [0, 4], [-1, 1]))
    mu1 = tf.slice(preds, [0, 0], [-1, 1])
    mu2 = tf.slice(preds, [0, 1], [-1, 1])
    # avoiding building a matrix...

    z = tf.pow(mu1 - tf.slice(y, [0, 0], [-1, 1]), 2.) * tf.exp(-log_s1) + \
        tf.pow(mu2 - tf.slice(y, [0, 1], [-1, 1]), 2.) * tf.exp(-log_s2) - \
        2 * rho * (mu1 - tf.slice(y, [0, 0], [-1, 1])) * (mu2 - tf.slice(y, [0, 1], [-1, 1])) * tf.sqrt(
        tf.exp(-log_s1) * tf.exp(-log_s2))

    return tf.reduce_mean(z / (2 * (1 - tf.pow(rho, 2.))) + 2 * np.pi * tf.sqrt(
        tf.exp(-log_s1) * tf.exp(-log_s2) * (1 - tf.pow(rho, 2.))))

def train(model_init_fn, optimizer_init_fn, cost_fn, data, device, fname,\
          restore = False,num_epochs = 1, print_every = 10, lr = 0.0005):
    tf.reset_default_graph()
    train_dset, val_dset, _ = data
    with tf.device(device):

        x = tf.placeholder(tf.float32, [None, 64,64,64,1])
        y = tf.placeholder(tf.float32, [None,2])

        training = tf.placeholder(tf.bool, name='training')

        preds = model_init_fn(x, training)
        #loss = tf.losses.absolute_difference(labels=y, predictions=preds, reduction=tf.losses.Reduction.SUM)
        loss = cost_fn(y, preds)

        #loss = tf.reduce_mean(loss)

        optimizer = optimizer_init_fn(lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

    with tf.device('/cpu:0'):

        saver = tf.train.Saver()

    with tf.Session() as sess:
        if restore:
            saver.restore(sess, fname)
        else:
            sess.run(tf.global_variables_initializer())
        t = 0
        for epoch in xrange(num_epochs):
            print 'Starting epoch %d' % epoch
            for x_np, y_np in train_dset:
                feed_dict = {x: x_np, y: y_np, training: True}
                #loss_np, update_ops_np = sess.run([loss,update_ops], feed_dict=feed_dict)
                loss_np, _  = sess.run([loss, train_op], feed_dict=feed_dict)

                if t % print_every == 0:
                    print 'Iteration %d, loss = %.4f' % (t, loss_np)
                    check_accuracy(sess, val_dset, x, preds, training=training)
                    print()
                    saver.save(sess, fname, global_step = t)
                t += 1

        saver.save(sess, fname, global_step = t) #save one last time

def test(model_init_fn, data,n_samples, device, fname, samples_fname_base):
    tf.reset_default_graph()
    with tf.device(device):

        x = tf.placeholder(tf.float32, [None, 64,64,64,1])

        training = tf.placeholder(tf.bool, name='training')

        preds = model_init_fn(x, training)

        saver = tf.train.Saver()


    with tf.Session() as sess:
        saver.restore(sess, fname)
        print 'Starting sampling'
        for x_np in data:
            samples = []
            feed_dict = {x: x_np, training: False}
            #loss_np, update_ops_np = sess.run([loss,update_ops], feed_dict=feed_dict)

            for i in xrange(n_samples):
                preds_np  = sess.run(preds, feed_dict=feed_dict)
                samples.append(preds_np)

            samples = np.vstack(samples)
            np.savetxt(samples_fname_base + '_%02d.npy'%i, samples)

def check_accuracy(sess, dset, x, scores, training=None):
    """
    Check accuracy on a classification model.

    Inputs:
    - sess: A TensorFlow Session that will be used to run the graph
    - dset: A Dataset object on which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.

    Returns: Nothing, but prints the accuracy of the model
    """
    perc_error = []
    do_chi2 = False
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, training: 0}
        y_pred = sess.run(scores, feed_dict=feed_dict)
        if y_pred.shape[1] == 2:
            perc_error.append((y_pred[:,:2]-y_batch)/(y_batch))
        else: # chi2
            do_chi2 = True
            mu1, mu2, log_s1, log_s2, rho = y_pred.T
            rho = np.tanh(rho)

            z = (mu1 - y_batch[:,0])**2 * np.exp(-log_s1) + \
                (mu2 - y_batch[:,1])**2 * np.exp(-log_s2) - \
                2 * rho * (mu1 - y_batch[:,0]) * (mu2 - y_batch[:,1]) * np.sqrt(np.exp(-log_s1) * np.exp(-log_s2))

            chi2= (z / (2 * (1 - rho**2.)) + 2 * np.pi * np.sqrt(
                np.exp(-log_s1) * np.exp(-log_s2) * (1 - rho**2.)))
            perc_error.append(np.mean(chi2))

    if not do_chi2:
        acc = np.array(perc_error[0]).mean(axis = 0)
        print 'Om: %.2f%%, s8: %.2f%% accuracy' % (100 * acc[0], 100*acc[1])
    else:
        print 'chi2: %.3f'%(np.mean(perc_error)/5)

"""
Train the neural network passed in.
Mostly copied from the CS231n notes
"""

try:
    import tensorflow as tf
except:
    pass
import numpy as np

def train(model_init_fn, optimizer_init_fn,data, device, fname, restore = False,num_epochs = 1, print_every = 10, lr = 0.0005):
    tf.reset_default_graph()
    train_dset, val_dset, _ = data
    with tf.device(device):

        x = tf.placeholder(tf.float32, [None, 64,64,64,1])
        y = tf.placeholder(tf.float32, [None,2])

        training = tf.placeholder(tf.bool, name='training')

        preds = model_init_fn(x, training)
        #loss = tf.losses.absolute_difference(labels=y, predictions=preds, reduction=tf.losses.Reduction.SUM)
        loss = tf.losses.mean_squared_error(labels=y, predictions=preds, reduction=tf.losses.Reduction.SUM)

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
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, training: 0}
        y_pred = sess.run(scores, feed_dict=feed_dict)
        
        print y_pred, '\n', y_batch#,'\n'#, (y_pred-y_batch)/y_batch
        print '*'*30
    #acc = float(num_correct) / num_samples
    #print 'Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc)

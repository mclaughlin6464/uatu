"""
Train the neural network passed in.
Mostly copied from the CS231n notes
"""

import tensorflow as tf

def train(model_init_fn, optimizer_init_fn,data, device, num_epochs = 1, print_every = 100):
    tf.reset_default_graph()
    train_dset, val_dset, _ = data
    with tf.device(device):

        x = tf.placeholder(tf.float32, [None, 64,64,64,1])
        y = tf.placeholder(tf.float32, [None,2])

        training = tf.placeholder(tf.bool, name='training')

        predictions = model_init_fn(x, training)

        loss = tf.losses.absolute_difference(labels=y, predictions=predictions)
        #loss = tf.reduce_mean(loss)

        optimizer = optimizer_init_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t = 0
        for epoch in xrange(num_epochs):
            print 'Starting epoch %d' % epoch
            for x_np, y_np in train_dset:
                feed_dict = {x: x_np, y: y_np, training: True}
                loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                if t % print_every == 0:
                    print 'Iteration %d, loss = %.4f' % (t, loss_np)
                    check_accuracy(sess, val_dset, x, predictions, training=training)
                    print()
                t += 1


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
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, training: 0}
        scores_np = sess.run(scores, feed_dict=feed_dict)
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    print 'Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc)
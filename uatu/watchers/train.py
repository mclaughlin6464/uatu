"""
Train the neural network passed in.
Mostly copied from the CS231n notes
"""

try:
    import tensorflow as tf
except:
    pass
import numpy as np
import sys

def standard_cost_fn(y, preds):
    return tf.losses.mean_squared_error(labels=y, predictions=preds, reduction=tf.losses.Reduction.SUM)

def bayes_cost_fn(y, preds):
    '''
    log_s1 = tf.slice(preds, [0, 2], [-1, 1])
    log_s2 = tf.slice(preds, [0, 3], [-1, 1])
    #rho = tf.tanh(tf.slice(preds, [0, 4], [-1, 1]))
    mu1 = tf.slice(preds, [0, 0], [-1, 1])
    mu2 = tf.slice(preds, [0, 1], [-1, 1])
    # avoiding building a matrix...

    z = tf.pow(mu1 - tf.slice(y, [0, 0], [-1, 1]), 2.) / (tf.exp(log_s1)+ 1e-3) + \
        tf.pow(mu2 - tf.slice(y, [0, 1], [-1, 1]), 2.) / (tf.exp(log_s2)+1e-3)# - \
        #2 * rho * (mu1 - tf.slice(y, [0, 0], [-1, 1])) * (mu2 - tf.slice(y, [0, 1], [-1, 1])) * tf.sqrt(
        #tf.exp(-log_s1) * tf.exp(-log_s2))

    return tf.reduce_mean(z +  log_s1 + log_s2), mu1, mu2, log_s1, log_s2, z#, rho
    #return tf.reduce_mean(z / (2 * (1 - tf.pow(rho, 2.))) + 2 * np.pi * tf.sqrt(
    #    tf.exp(log_s1) * tf.exp(log_s2) * (1 - tf.pow(rho, 2.)))), mu1, mu2, log_s1, log_s2, rho
    '''
    num_out = 2 
    s = tf.slice(preds , [0,num_out/2] , [-1,num_out/2] )
    y_conv = tf.slice(preds , [0,0] , [-1,num_out/2] )
    return tf.reduce_mean( tf.pow( y_conv -  y , 2.) * tf.exp(-s) + s)#  ,axis=1)# , [-1 , 1 ])

def original_bayes_cost_fn(y, preds):
    log_s1 = tf.slice(preds, [0, 2], [-1, 1])
    log_s2 = tf.slice(preds, [0, 3], [-1, 1])
    rho = 0.9*tf.tanh(tf.slice(preds, [0, 4], [-1, 1]))
    mu1 = tf.slice(preds, [0, 0], [-1, 1])
    mu2 = tf.slice(preds, [0, 1], [-1, 1])
    # avoiding building a matrix...

    z = tf.pow(mu1 - tf.slice(y, [0, 0], [-1, 1]), 2.) / (tf.exp(log_s1)+ 1e-3) + \
        tf.pow(mu2 - tf.slice(y, [0, 1], [-1, 1]), 2.) / (tf.exp(log_s2)+1e-3) - \
        2 * rho * (mu1 - tf.slice(y, [0, 0], [-1, 1])) * (mu2 - tf.slice(y, [0, 1], [-1, 1])) / tf.sqrt(
        tf.exp(log_s1) * tf.exp(log_s2)+1e-6)

    return tf.reduce_mean(z / (2.0 * (1 - tf.pow(rho, 2.))) + 
        (log_s1 + log_s2 + tf.log(1 - tf.pow(rho, 2.)))/2.0 )


def train(model_init_fn, optimizer_init_fn, cost_fn, data, device, fname,\
          restore = False,num_epochs = 1, print_every = 10):
    tf.reset_default_graph()
    train_dset, val_dset, _ = data
    with tf.device(device):

        x = tf.placeholder(tf.float32, [None, 64,64,64,1])
        y = tf.placeholder(tf.float32, [None,2])

        training = tf.placeholder(tf.bool, name='training')

        preds = model_init_fn(x, training=training)
        #loss = tf.losses.absolute_difference(labels=y, predictions=preds, reduction=tf.losses.Reduction.SUM)
        #loss, mu1, mu2, log_s1, log_s2, z = cost_fn(y, preds)
        loss = cost_fn(y, preds)

        #loss = tf.reduce_mean(loss)

        lr = tf.placeholder(tf.float32, name = 'learning_rate')
        optimizer = optimizer_init_fn(lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #train_op = optimizer.minimize(loss)
            gvs = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs)

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
            sys.stdout.flush()
            for x_np, y_np in train_dset:
                print t,
                sys.stdout.flush()

                if epoch < 3:
                    lr_np = 5e-5 
                else:
                    lr_np = 5e-7
                feed_dict = {x: x_np, y: y_np, training: True, lr: lr_np}
                #loss_np, update_ops_np = sess.run([loss,update_ops], feed_dict=feed_dict)
                #loss_np,mu1_np,mu2_np, log_s1_np, log_s2_np, z_np, _  = sess.run([loss,mu1,mu2,log_s1, log_s2, z, train_op], feed_dict=feed_dict)
                loss_np, _  = sess.run([loss, train_op], feed_dict=feed_dict)

                print loss_np#,mu1_np, mu2_np, np.exp(log_s1_np), np.exp(log_s2_np), z_np

                if t % print_every == 0:
                    print 'Iteration %d, loss = %.4f' % (t, loss_np)
                    sys.stdout.flush() 
                    check_accuracy(sess, val_dset, x, preds, training=training)
                    print
                    sys.stdout.flush()
                    saver.save(sess, fname, global_step = t)
                t += 1
                sys.stdout.flush()

        saver.save(sess, fname, global_step = t) #save one last time

def test(model_init_fn, data,n_samples, device, fname, samples_fname_base):
    tf.reset_default_graph()
    with tf.device(device):

        x = tf.placeholder(tf.float32, [None, 64,64,64,1])

        training = tf.placeholder(tf.bool, name='training')

        preds = model_init_fn(x, training)

    with tf.device('/cpu:0'):

        saver = tf.train.Saver()


    with tf.Session() as sess:
        saver.restore(sess, fname)
        print 'Starting sampling'
        for i, (x_np,  y_np) in enumerate(data):
            samples = []
            feed_dict = {x: x_np, training: False}
            #loss_np, update_ops_np = sess.run([loss,update_ops], feed_dict=feed_dict)
            y1, y2 = y_np[0]
            for j in xrange(n_samples):
                preds_np  = sess.run(preds, feed_dict=feed_dict)
                samples.append(preds_np)

            samples = np.vstack(samples)

            print samples.mean(axis = 0)
            print samples.std(axis = 0)
            print  y1, y2
            np.savetxt(samples_fname_base + '_%02d.npy'%i, samples, header = '%.6f\t%.6f'%(y1, y2))

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
            if y_pred.shape[1] == 4:
                mu1, mu2, log_s1, log_s2 = y_pred.T
                rho = 0
            else:
                mu1, mu2, log_s1, log_s2, rho = y_pred.T

            rho = 0.9*np.tanh(rho)

            chi2 = (mu1 - y_batch[:,0])**2 /(np.exp(log_s1)+1e-3) + \
                (mu2 - y_batch[:,1])**2 /(np.exp(log_s2)+1e-3) - \
                2 * rho * (mu1 - y_batch[:,0]) * (mu2 - y_batch[:,1]) / np.sqrt(np.exp(log_s1) * np.exp(log_s2)+1e-6)

            #chi2= (z / (2 * (1 - rho**2.)) + 
            #    log_s1 + log_s2 + np.log(1 - rho**2.))
            perc_error.append(np.mean(chi2))

    if not do_chi2:
        acc = np.abs(np.array(perc_error[0]).mean(axis = 0))
        print 'Om: %.2f%%, s8: %.2f%% accuracy' % (100 * acc[0], 100*acc[1])
    else:
        print 'chi2: %.3f'%(np.mean(perc_error)/2)

"""
Perform adversarial training against NNs
"""
try:
    import tensorflow as tf
except ImportError:
    pass
from functools import wraps

from .train import standard_abs_cost_fn, standard_cost_fn, bayes_cost_fn, original_bayes_cost_fn
from __future__ import  print_function


def adversarial(epsilon = 0.05, K=1):
    """
    Decorator to convert loss functions into adversarial training loss functions
    :param epsilon:
        Radius of the norm-ball within which the adversarial examples live. Default 0.01
    :param K:
        Number of gradient steps to perform in the construction of adversarial examples. Default 10.
    :return:
        func, decorated function
    """
    def actual_decorator(loss_func):
        @wraps(loss_func)
        def _func(y, preds, x, model, model_kwargs, scope):

            loss = loss_func(y, preds)
            grads = tf.gradients(loss, x)

            dx = tf.clip_by_value(tf.sign(grads), -epsilon, epsilon)[0]

            x+=dx
            scope.reuse_variables()
            adv_preds = model(x, **model_kwargs)
            tf.stop_gradient(adv_preds)

            for i in range(K-1):
                #print(i)
                loss = loss_func(y, adv_preds)
                grads = tf.gradients(loss, x)

                dx = tf.clip_by_value(tf.sign(grads), -epsilon, epsilon)[0]

                x+=dx
                adv_preds = model(x, **model_kwargs)
                tf.stop_gradient(adv_preds)

            return 0.5*loss + 0.5*loss_func(y, adv_preds)

        return _func

    return actual_decorator

@adversarial()
def adversarial_standard_cost_fn(y,preds):
    return standard_cost_fn(y, preds)

@adversarial()
def adversarial_standard_abs_cost_fn(y,preds):
    return standard_abs_cost_fn(y, preds)

@adversarial()
def adversarial_bayes_cost_fn(y,preds):
    return bayes_cost_fn(y, preds)

@adversarial()
def adversarial_original_bayes_cost_fn(y, preds):
    return original_bayes_cost_fn(y, preds)

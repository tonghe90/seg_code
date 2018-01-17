import tensorflow as tf
#import glog as log
# import ipdb
import numpy as np
import os
slim = tf.contrib.slim



class DataSource:
    def next(self):
        assert False

class SolverParameter:
    def __init__(self):
        ### lr
        self.type = None
        self.base_lr = 0.5
        self.lr_policy = None
        self.gamma = 0.1
        self.stepsize = 5000

        self.exclude_scope = ''
        self.gpu_list = '0'
        self.gpu_ids = self.get_gpu_ids()
    def get_gpu_ids(self):
        return [int(_) for _ in self.gpu_list.strip().split(',')]


def get_learning_rate(lr_policy, base_lr, global_step, gamma, step_value):
    if lr_policy == 'fixed':
        return tf.constant(base_lr, name='fixed_lr')
    elif lr_policy == 'step':
        return tf.train.exponential_decay(learning_rate=base_lr,
                                               global_step=global_step,
                                               decay_rate=gamma,
                                               decay_steps=step_value,
                                               staircase=True,
                                               name='step_lr')
    elif lr_policy == 'exponential':
        return tf.train.exponential_decay(learning_rate=base_lr,
                                               global_step=global_step,
                                               decay_rate=gamma,
                                               decay_steps=step_value,
                                               name='exp_lr')
    else:
        raise ValueError("No such learning policy --- {}".format(lr_policy))


def get_optimizer(solver_type, learning_rate):
    if solver_type == "SGD":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif solver_type == 'ADAM':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif solver_type == 'ADADELTA':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    # elif solver_type == 'MOMENTUM':
    #     optimizer = tf.train.MomentumOptimizer(learning_rate)
    else:
        raise ValueError("No such solver type --- {}".format(solver_type))
    return optimizer

def get_device_names(solver_param):
    if solver_param.gpu_list:
        gpus = [int(_) for _ in solver_param.gpu_list.split(',')]
        return ['/device:GPU:%d' % int(g) for g in gpus]
    else:
        return ['/device:GPU:0']

def _gather_clone_loss(clone, num_clones, regularization_losses):
    sum_loss = None
    # Individual components of the loss that will need summaries.
    clone_loss = None
    regularization_loss = None
    # Compute and aggregate losses on the clone device.
    with tf.device(clone['device']):
        all_losses = []
        clone_losses = tf.get_collection(tf.GraphKeys.LOSSES, clone['scope'])
        if clone_losses:
            clone_loss = tf.add_n(clone_losses, name='clone_loss')
            if num_clones > 1:
                clone_loss = tf.div(clone_loss, 1.0 * num_clones,
                                    name='scaled_clone_loss')
            all_losses.append(clone_loss)
        if regularization_losses:
            regularization_loss = tf.add_n(regularization_losses,
                                           name='regularization_loss')
            all_losses.append(regularization_loss)
        if all_losses:
            sum_loss = tf.add_n(all_losses)
    # Add the summaries out of the clone device block.
    # if clone_loss is not None:
    #     tf.summary.scalar(clone['scope'] + '/clone_loss', clone_loss)
    # if regularization_loss is not None:
    #     tf.summary.scalar('regularization_loss', regularization_loss)
    #

    return sum_loss

def _optimize_clone(optimizer, clone, num_clones, regularization_losses):
    sum_loss = _gather_clone_loss(clone, num_clones, regularization_losses)
    clone_grad = None
    if sum_loss is not None:
        with tf.device(clone['device']):
            clone_grad = optimizer.compute_gradients(sum_loss)
    return sum_loss, clone_grad

def _sum_clones_gradients(clone_grads):
    """Calculate the sum gradient for each shared variable across all clones.
    This function assumes that the clone_grads has been scaled appropriately by
    1 / num_clones.
    Args:
      clone_grads: A List of List of tuples (gradient, variable), one list per
      `Clone`.
    Returns:
       List of tuples of (gradient, variable) where the gradient has been summed
       across all clones.
    """
    sum_grads = []
    for grad_and_vars in zip(*clone_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))
        grads = []
        var = grad_and_vars[0][1]
        for g, v in grad_and_vars:
            assert v == var
            if g is not None:
              grads.append(g)
        if grads:
            if len(grads) > 1:
              sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
            else:
              sum_grad = grads[0]
            sum_grads.append((sum_grad, var))
    return sum_grads


def _summary(loss):
    keys = loss.keys()
    for n in range(len(keys)):
        tf.summary.scalar(keys[n], loss[keys[n]])




class Solver:
    def __init__(self, solver_param, sess):
        self.solver_param = solver_param
        self.sess = sess
        self.graph = self.sess.graph
        assert self.graph == tf.get_default_graph()

        self.global_step = tf.train.get_or_create_global_step(graph=self.graph)
        self.lr = get_learning_rate(lr_policy=solver_param.lr_policy,
                                    base_lr=solver_param.base_lr,
                                    gamma=solver_param.gamma,
                                    global_step=self.global_step,
                                    step_value=solver_param.stepsize)
        tf.summary.scalar('learning_rate', self.lr)
        self.optimizer = get_optimizer(solver_param.type, self.lr)
        #self.device_names = get_device_names(solver_param)
        gpus = [int(_) for _ in solver_param.gpu_list.split(',')]
        self.num_gpus = len(gpus)


    @property
    def learning_rate(self):
        return self.lr

    def create_clones(self, model_fn, loss_fn, eval_metric, split_inputs, split_targets):
        clones = []
        print('gpus num:', self.num_gpus)
        for idx in range(self.num_gpus):
            with tf.name_scope('tower_{}'.format(idx)) as scope:
                clone_device = '/gpu:{}'.format(idx)
                with tf.device(clone_device):
                    with tf.variable_scope(tf.get_variable_scope(),
                                               reuse=(idx>0)):

                        infers = model_fn(inputs=split_inputs[idx])

                        loss = loss_fn(infers=infers, targets=split_targets[idx])
                        acc = eval_metric(infers=infers, targets=split_targets[idx])

            clones.append({'metric': acc, 'scope': scope, 'device': clone_device})

        return clones


    def deploy(self, model_fn, loss_fn, eval_fn, inputs, targets, total_size, regularize=False):
        grads_and_vars = []
        clones_losses = []

        split_size = [int(total_size / self.num_gpus) for _ in range(self.num_gpus)]
        if total_size % self.num_gpus != 0:
            split_size[-1] += (total_size % self.num_gpus)

        split_inputs = [{} for _ in range(self.num_gpus)]
        for k in inputs:
            splits = tf.split(inputs[k], split_size, axis=0)
            for i in range(self.num_gpus):
                split_inputs[i][k] = splits[i]



        split_targets = [{} for _ in range(self.num_gpus)]
        for k in targets:
            splits = tf.split(targets[k], split_size, axis=0)
            for i in range(self.num_gpus):
                split_targets[i][k] = splits[i]

        clones = self.create_clones(model_fn, loss_fn, eval_fn, split_inputs, split_targets)

        num_clones = len(clones)
        regularization_losses = None
        if regularization_losses is None and regularize:
            regularization_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)

        for clone in clones:
            with tf.name_scope(clone['scope']):
                clone_loss, clone_grad = _optimize_clone(self.optimizer, clone, num_clones, regularization_losses)

            if clone_loss is not None:
                clones_losses.append(clone_loss)
                grads_and_vars.append(clone_grad)

            regularization_losses = None

        total_loss = tf.add_n(clones_losses, name='total_loss')
        grads_and_vars = _sum_clones_gradients(grads_and_vars)


        grad_updates = self.optimizer.apply_gradients(grads_and_vars,
                                                 global_step=self.global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, clones[0]['scope'])
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)

        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

        return train_tensor, total_loss, clones[0]['metric']


    def get_variables_continue_training(self):
        return tf.global_variables()

    def get_variables_finetune(self):
        if self.solver_param.exclude_scope == None:
            return tf.trainable_variables()
        exclusions = [scope.strip()
                      for scope in self.solver_param.exclude_scope.split(',')]
        variables_to_restore = []
        for var in tf.trainable_variables():
            excluded = False
            for exclusion in exclusions:
                if exclusion in var.name:
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
        return variables_to_restore


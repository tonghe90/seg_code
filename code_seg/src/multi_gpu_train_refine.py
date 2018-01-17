import tensorflow as tf
import cv2
import cfg
slim = tf.contrib.slim
import argparse
import os
from data_provider import COCO_detection_train, MultiDataProvider
from solvers import *
from tools import save, Timer
from segnas_refine import nas_refine_model
from enum import Enum

STATUS = Enum('STATUS', ('NO_INIT', 'FINTTUNE', 'CONTINUE'))


### param for data
BATCH_SIZE = 1
NUM_CLASSES = 80
ROOT_DIR = '/home/tonghe/Downloads/coco'
JSON_FILE_TRAIN = 'instances_train2017.json'
JSON_FILE_VAL = 'instances_val2017.json'
DATA_PROVIDER = 'coco'
CROP_SIZE = 512
MINING_RATIO = 0.5

### param for init weights
SOLVER_STATE = None
INIT_WEIGHTS = None

### param for solver
TYPE = 'ADAM'
LEARNING_RATE = 0.01
LR_POLICY = 'step'
GAMMA = 0.1
STEPSIZE = 100000
GPUS_LIST = '0'
EXCLUDE_SCOPE = None


###
SAVE_STEP = 8000
VAL_STEPS = 1000
SHOW_STEPS = 10
NUM_STEPS = 300000
OHEM = True
SAVE_DIR = '/home/tonghe/tmp'


###
scale = 0.25
min_ratio = 0.8
max_ratio = 1.3
hor_flip = True
test_iter = 100

def get_arguments():
    """
    Parse all the arguments provided from the CLI.
    Returns:
        A list of parsed arguments.
    """

    parser = argparse.ArgumentParser(description="ResNet-101-RefineNet Network")

    ### params for data
    parser.add_argument("--root-dir", type=str, default=ROOT_DIR,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--json-file-train", type=str, default=JSON_FILE_TRAIN,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--json-file-val", type=str, default=JSON_FILE_VAL,
                        help="Path to the file listing the images in the val dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--data-provider", type=str, default=DATA_PROVIDER,
                        help="Path to the file listing the images in the val dataset.")
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE,
                        help="Size for input images")
    parser.add_argument("--mining-ratio", type=float, default=MINING_RATIO,
                        help="ratio for hard mining")


    ### param for init weights
    parser.add_argument("--solver-state", type=str, default=SOLVER_STATE,
                        help="model file for fine tuning")
    parser.add_argument("--init-weights", type=str, default=INIT_WEIGHTS,
                        help="model file for init")

    ### param for solver
    parser.add_argument("--type", type=str, default=TYPE,
                        help="type of solver")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate.")
    parser.add_argument("--lr-policy", type=str, default=LR_POLICY,
                        help="learning policy")
    parser.add_argument("--step-size", type=int, default=STEPSIZE,
                        help="step size for solver")
    parser.add_argument("--gpus-list", type=str, default=GPUS_LIST,
                        help="gpus")
    parser.add_argument("--exclude-scope", type=str, default=EXCLUDE_SCOPE,
                        help="learning policy")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                        help="gamma for learning rate")

    ###
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Total Number of training steps.")
    parser.add_argument("--show-steps", type=int, default=SHOW_STEPS,
                        help="Show results in every show-steps.")
    parser.add_argument("--val-steps", type=int, default=VAL_STEPS,
                        help="Validation steps")
    parser.add_argument("--ohem", type=bool, default=OHEM,
                        help="Whether conduct ohem during the training process")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="dir for saving models and logs")

    parser.add_argument("--save-step", type=int, default=SAVE_STEP,
                        help="the frequence of saving model")



    return parser.parse_args()


def train():
    args = get_arguments()
    print '*' * 10 + ' args ' + '*' * 10
    print args
    print '*' * 26
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.data_provider == "coco":
        pass
        coco_provider_train = COCO_detection_train(args.root_dir, args.json_file_train)
        coco_provider_val = COCO_detection_train(args.root_dir, args.json_file_val)

        data_provider_train = MultiDataProvider([coco_provider_train], crop_size=args.crop_size,
                                                min_ratio=min_ratio, max_ratio=max_ratio, hor_flip=hor_flip)

        data_provider_val = MultiDataProvider([coco_provider_val], crop_size=args.crop_size,
                                              min_ratio=1, max_ratio=1.1, hor_flip=False)


    else:
        raise RuntimeError, 'unknown data provider type'


    solver_param = SolverParameter()
    solver_param.type = args.type
    solver_param.base_lr = args.learning_rate
    solver_param.lr_policy = args.lr_policy
    solver_param.gamma = args.gamma
    solver_param.stepsize = args.step_size
    solver_param.gpu_list = args.gpus_list
    solver_param.exclude_scope = args.exclude_scope


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    solver = Solver(solver_param, sess)

    total_size = args.batch_size *solver.num_gpus
    print('total size for single forward: ', total_size)


    with tf.name_scope('input'):
        inputs = {'images': tf.placeholder(tf.float32,
                                           shape=(total_size, args.crop_size, args.crop_size, 3),
                                           name='input_images')}
        label_gt = {'labels': tf.placeholder(dtype=tf.int32,
                                  shape=(total_size, int(args.crop_size * scale), int(args.crop_size * scale)),
                                  name='input_label')}

    global_step = tf.train.get_or_create_global_step()
    nas_refine_net = nas_refine_model(num_cls=args.num_classes,
                                      is_training=True, ohem=args.ohem, mining_ratio=args.mining_ratio)


    train_op, total_loss, metric = solver.deploy(model_fn=nas_refine_net.infer,
                                                 loss_fn=nas_refine_net.loss_fn,
                                                 eval_fn=nas_refine_net.eval_metric,
                                                 inputs=inputs, targets=label_gt,
                                                 total_size=total_size, regularize=True)

    miou = metric['miou']
    tf.summary.scalar('miou', miou)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('lr', solver.learning_rate)

    ### init
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    ### restore
    flag = STATUS.NO_INIT
    variables_to_restore = None
    if args.init_weights != None:
        flag = STATUS.FINTTUNE
        # if not os.path.exists(args.init_weights):
        #     raise RuntimeError, '{} does not exist for finetuning'.format(args.init_weights)
        init_weights = args.init_weights
        variables_to_restore = solver.get_variables_finetune()

    if args.solver_state != None:
        flag = STATUS.CONTINUE
        if os.path.isdir(args.solver_state):
            solver_state = tf.train.latest_checkpoint(args.solver_state)
        else:
            # if not os.path.exists(args.solver_state):
            #     raise RuntimeError, '{} does not exist for continue training'.format(args.solver_state, flag)
            solver_state = args.solver_state
        variables_to_restore = solver.get_variables_continue_training()


    if flag == STATUS.FINTTUNE:
        loader = tf.train.Saver(var_list=variables_to_restore)
        loader.restore(sess, init_weights)
        print('{} loaded'.format(init_weights))
    elif flag == STATUS.CONTINUE:
        loader = tf.train.Saver(var_list=variables_to_restore)
        loader.restore(sess, solver_state)
        print('{} loaded'.format(solver_state))

    all_summaries = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.save_dir)
    saver = tf.train.Saver()

    label_sz = int(args.crop_size*scale)
    #timer = Timer()
    for step in range(1, args.num_steps):
        #print 'step: ', step
        #data_blob = np.zeros((total_size, args.crop_size, args.crop_size, 3), dtype=np.float32)
        #label_blob = np.zeros((total_size, label_sz, label_sz), dtype=np.int32)
	data_blob = np.zeros((0, args.crop_size, args.crop_size, 3), dtype=np.float32)
        label_blob = np.zeros((0, label_sz, label_sz), dtype=np.int32)
	timer = Timer()
	timer.tic()
	for cur_id in range(solver.num_gpus):
            #start = int(cur_id * args.batch_size)
            #end = int((cur_id + 1) * args.batch_size)
            while True:
                images, labels = data_provider_train.get_batch(args.batch_size)
                #data_blob[start:end] = images
                labels_resize = np.zeros((labels.shape[0], label_sz, label_sz), dtype=np.int32)
                for n in range(labels.shape[0]):
                    labels_resize[n] = cv2.resize(labels[n], (label_sz, label_sz),
                                                      interpolation=cv2.INTER_NEAREST)

                #label_blob[start:end] = labels_resize

                if np.any(labels_resize >= 0):
                    data_blob = np.concatenate((data_blob, images), axis=0)
                    label_blob = np.concatenate((label_blob, labels_resize), axis=0)
                    break

        assert label_blob.shape[0] == total_size
        assert data_blob.shape[0] == total_size

        ### prepare data
        #for cur_id in range(solver.num_gpus):
        #    start = int(cur_id * args.batch_size)
        #    end = int((cur_id + 1) * args.batch_size)
        #    while True:
        #        images, labels = data_provider_train.get_batch(args.batch_size)
        #        data_blob[start:end] = images
        #        labels_resize = np.zeros((labels.shape[0], label_sz, label_sz), dtype=np.int32)
        #        for n in range(labels.shape[0]):
        #            labels_resize[n] = cv2.resize(labels[n], (label_sz, label_sz),
        #                                              interpolation=cv2.INTER_NEAREST)

        #        label_blob[start:end] = labels_resize
        #        if np.any(labels_resize >= 0):
        #            break


        ### run training op
        _, losses_value, _, summary, global_step_val = sess.run([train_op, total_loss, metric['op'], all_summaries, global_step],
                 feed_dict={inputs['images']: data_blob, label_gt['labels']: label_blob})

        summary_writer.add_summary(summary, global_step=global_step_val)

        ### show
        if step % args.show_steps == 0:
	    t1 = timer.toc()
            print('step: {}, lr: {}, loss_value: {}, miou: {}, time: {} / per iter'.format(global_step_val, sess.run(solver.learning_rate),
                                                            losses_value, miou.eval(session=sess), t1))
	     
	    #time = timer.toc()
	    
        ## save
        if step % args.save_step == 0:
            save(saver, sess, args.save_dir, step=global_step_val)

        ### test
        if step % args.val_steps == 0:
	    test_loss = 0
            print('#' * 5 + ' testing ' + '#' * 5)
            for kk in range(test_iter):
                for cur_id in range(solver.num_gpus):
                    start = int(cur_id * args.batch_size)
                    end = int((cur_id + 1) * args.batch_size)
                    while True:
                        images, labels = data_provider_val.get_batch(args.batch_size)

                        data_blob[start:end] = images
                        labels_resize = np.zeros((labels.shape[0], label_sz, label_sz), dtype=np.int32)
                        for n in range(labels.shape[0]):
                            labels_resize[n] = cv2.resize(labels[n], (label_sz, label_sz),
                                                              interpolation=cv2.INTER_NEAREST)
                        label_blob[start:end] = labels_resize
                        if np.any(labels_resize >= 0):
                            break

                losses_value = sess.run([total_loss],
                                    feed_dict={inputs['images']: data_blob, label_gt['labels']: label_blob})
                test_loss += losses_value[0]
                
            print('global_step_val: {}, test loss: {}'.format(global_step_val, float(test_loss)/test_iter))
            print('#' * 19)









if __name__ == '__main__':
    train()



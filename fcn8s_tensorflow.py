from distutils.version import LooseVersion
import tensorflow as tf
import warnings
from tqdm import trange
import sys
import os.path
import scipy.misc
import shutil
from glob import glob
from collections import deque
import numpy as np
import time

from tf_variable_summaries import add_variable_summaries
from visualization_utils import print_segmentation_onto_image

class FCN8s:

    def __init__(self, model_load_dir=None, tags=None, vgg16_dir=None, num_classes=None):
        '''
        Arguments:
            vgg16_dir (string): The directory that contains the pretrained VGG-16 model. The directory needs to
                                contain a `variables/` directory that contains the variables checkpoint, and a
                                "saved_model.pb" protocol buffer.
        '''
        # Check TensorFlow version
        assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'This program requires TensorFlow version 1.0 or newer. You are using {}'.format(tf.__version__)
        print('TensorFlow Version: {}'.format(tf.__version__))

        if (model_load_dir is None) and (vgg16_dir is None or num_classes is None):
            raise ValueError("You must provide either both `model_load_dir` and `tags` or both `vgg16_dir` and `num_classes`.")

        self.model_load_dir = model_load_dir
        self.tags = tags
        self.vgg16_dir = vgg16_dir
        self.vgg16_tag = 'vgg16'
        self.num_classes = num_classes

        self.variables_updated = False # Keep track of whether any variable values changed since this model was last saved.
        self.eval_dataset = None # Which dataset to use for evaluation during training. Only relevant for training.

        # The following lists store data about the metrics being tracked.
        # Note that `self.metric_value_tensors` and `self.metric_update_ops` represent
        # the metrics being tracked, not the metrics generally available in the model.
        self.metric_names = [] # Store the metric names here.
        self.metric_values = [] # Store the latest metric evaluations here.
        self.best_metric_values = [] # Keep score of the best historical metric values.
        self.metric_value_tensors = [] # Store the value tensors from tf.metrics here.
        self.metric_update_ops = [] # Store the update ops from tf.metrics here.

        self.training_loss = None
        self.best_training_loss = 99999999.9

        self.sess = tf.Session()
        self.g_step = None # The global step

        ##################################################################
        # Load or build the model.
        ##################################################################

        if not model_load_dir is None: # Load the full pre-trained model.

            tf.saved_model.loader.load(sess=self.sess, tags=self.tags, export_dir=self.model_load_dir)
            graph = tf.get_default_graph()

            # Get the input and output ops.
            self.image_input = graph.get_tensor_by_name('image_input:0')
            self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
            self.fcn8s_output = graph.get_tensor_by_name('decoder/fcn8s_output:0')
            self.labels = graph.get_tensor_by_name('labels_input:0')
            self.loss = graph.get_tensor_by_name('optimizer/loss:0')
            self.train_op = graph.get_tensor_by_name('optimizer/train_op:0')
            self.learning_rate = graph.get_tensor_by_name('optimizer/learning_rate:0')
            self.global_step = graph.get_tensor_by_name('optimizer/global_step:0')
            self.softmax_output = graph.get_tensor_by_name('predictor/softmax_output:0')
            self.mean_loss_value = graph.get_tensor_by_name('metrics/mean_loss_value:0')
            self.mean_loss_update_op = graph.get_tensor_by_name('metrics/mean_loss_update_op:0')
            self.mean_iou_value = graph.get_tensor_by_name('metrics/mean_iou_value:0')
            self.mean_iou_update_op = graph.get_tensor_by_name('metrics/mean_iou_update_op:0')
            self.acc_value = graph.get_tensor_by_name('metrics/acc_value:0')
            self.acc_update_op = graph.get_tensor_by_name('metrics/acc_update_op:0')
            self.metrics_reset_op = graph.get_operation_by_name('metrics/metrics_reset_op')
            self.summaries_training = graph.get_tensor_by_name('summaries_training:0')
            self.summaries_evaluation = graph.get_tensor_by_name('summaries_evaluation:0')

            # For some reason that I don't understand, the local variables belonging to the
            # metrics need to be initialized after loading the model.
            self.sess.run(self.metrics_reset_op)

        else: # Load only the pre-trained VGG-16 encoder and build the rest of the graph from scratch.

            # Load the pretrained convolutionalized VGG-16 model as our encoder.
            self.image_input, self.keep_prob, self.pool3_out, self.pool4_out, self.fc7_out = self._load_vgg16()
            # Build the decoder on top of the VGG-16 encoder.
            self.fcn8s_output = self._build_decoder()
            # Build the part of the graph that is relevant for the training.
            self.labels = tf.placeholder(dtype=tf.int32, shape=[None, None, None, self.num_classes], name='labels_input')
            self.loss, self.train_op, self.learning_rate, self.global_step = self._build_optimizer()
            # Add the evaluator.
            self.softmax_output = self._build_predictor()
            self.mean_loss_value, self.mean_loss_update_op, self.mean_iou_value, self.mean_iou_update_op, self.acc_value, self.acc_update_op, self.metrics_reset_op = self._build_metrics()
            # Add summary ops.
            self.summaries_training, self.summaries_evaluation = self._build_summary_ops()
            # Initialize the global and local (for the metrics) variables.
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

    def _load_vgg16(self):
        '''
        Loads the pretrained VGG-16 model into `sess`.

            vgg_path (string): The directory that contains the pretrained VGG-16 model. The directory needs to
                               contain a `variables/` directory that contains the variables checkpoint, and a
                               "saved_model.pb" protocol buffer.

        Return:
            A tuple of the five tensors that are relevant to build our FCN-8s on top of the VGG-16,
            namely `(image_input, keep_prob, layer3_out, layer4_out, layer7_out)`.
        '''

        # 1: Load the model

        tf.saved_model.loader.load(sess=self.sess, tags=[self.vgg16_tag], export_dir=self.vgg16_dir)

        # 2: Return the tensors of interest

        graph = tf.get_default_graph()

        vgg16_image_input_tensor_name = 'image_input:0'
        vgg16_keep_prob_tensor_name = 'keep_prob:0'
        vgg16_pool3_out_tensor_name = 'layer3_out:0'
        vgg16_pool4_out_tensor_name = 'layer4_out:0'
        vgg16_fc7_out_tensor_name = 'layer7_out:0'

        image_input = graph.get_tensor_by_name(vgg16_image_input_tensor_name)
        keep_prob = graph.get_tensor_by_name(vgg16_keep_prob_tensor_name)
        pool3_out = graph.get_tensor_by_name(vgg16_pool3_out_tensor_name)
        pool4_out = graph.get_tensor_by_name(vgg16_pool4_out_tensor_name)
        fc7_out = graph.get_tensor_by_name(vgg16_fc7_out_tensor_name)

        return image_input, keep_prob, pool3_out, pool4_out, fc7_out

    def _build_decoder(self):
        '''
        Builds the FCN-8s decoder given the pool3, pool4, and fc7 outputs of the VGG-16 encoder.

        Returns:
            The raw (i.e. not yet scaled by softmax) output of the final FCN-8s layer.
            This is a 4D tensor of shape `[batch, img_height, img_width, num_classes]`, i.e.
            the spatial dimensions of the output are the same as those of the input images.
        '''

        stddev_1x1 = 0.001 # Standard deviation for the 1x1 kernel initializers
        stddev_conv2d_trans = 0.01 # Standard deviation for the convolution transpose kernel initializers
        l2reg = 0.001 # L2 regularization rate for the kernels

        with tf.name_scope('decoder'):

            # 1: Append 1x1 convolutions to the three output layers of the encoder to reduce the Number
            #    of channels to the number of classes.

            # The outputs of pool3 and pool4 are being scaled in what the authors of
            # the paper call the at-once training approach.
            pool3_out_scaled = tf.multiply(self.pool3_out, 0.0001, name='pool3_out_scaled')

            pool3_1x1 = tf.layers.conv2d(inputs=pool3_out_scaled,
                                         filters=self.num_classes,
                                         kernel_size=(1, 1),
                                         strides=(1, 1),
                                         padding='same',
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=stddev_1x1),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(l2reg),
                                         name='pool3_1x1')

            pool4_out_scaled = tf.multiply(self.pool4_out, 0.01, name='pool4_out_scaled')

            pool4_1x1 = tf.layers.conv2d(inputs=pool4_out_scaled,
                                         filters=self.num_classes,
                                         kernel_size=(1, 1),
                                         strides=(1, 1),
                                         padding='same',
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=stddev_1x1),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(l2reg),
                                         name='pool4_1x1')

            fc7_1x1 = tf.layers.conv2d(inputs=self.fc7_out,
                                       filters=self.num_classes,
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding='same',
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=stddev_1x1),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(l2reg),
                                       name='fc7_1x1')

            # 2: Upscale and fuse until we're back at the original image size.

            fc7_conv2d_trans = tf.layers.conv2d_transpose(inputs=fc7_1x1,
                                                          filters=self.num_classes,
                                                          kernel_size=(4, 4),
                                                          strides=(2, 2),
                                                          padding='same',
                                                          kernel_initializer=tf.truncated_normal_initializer(stddev=stddev_conv2d_trans),
                                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(l2reg),
                                                          name='fc7_conv2d_trans')

            add_fc7_pool4 = tf.add(fc7_conv2d_trans, pool4_1x1, name='add_fc7_pool4')

            fc7_pool4_conv2d_trans = tf.layers.conv2d_transpose(inputs=add_fc7_pool4,
                                                                filters=self.num_classes,
                                                                kernel_size=(4, 4),
                                                                strides=(2, 2),
                                                                padding='same',
                                                                kernel_initializer=tf.truncated_normal_initializer(stddev=stddev_conv2d_trans),
                                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(l2reg),
                                                                name='fc7_pool4_conv2d_trans')

            add_fc7_pool4_pool3 = tf.add(fc7_pool4_conv2d_trans, pool3_1x1, name='add_fc7_pool4_pool3')

            fc7_pool4_pool3_conv2d_trans = tf.layers.conv2d_transpose(inputs=add_fc7_pool4_pool3,
                                                                      filters=self.num_classes,
                                                                      kernel_size=(16, 16),
                                                                      strides=(8, 8),
                                                                      padding='same',
                                                                      kernel_initializer=tf.truncated_normal_initializer(stddev=stddev_conv2d_trans),
                                                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(l2reg),
                                                                      name='fc7_pool4_pool3_conv2d_trans')

            fcn8s_output = tf.identity(fc7_pool4_pool3_conv2d_trans, name='fcn8s_output')

        return fc7_pool4_pool3_conv2d_trans

    def _build_optimizer(self):

        with tf.name_scope('optimizer'):
            # Create a training step counter.
            global_step = tf.Variable(0, trainable=False, name='global_step')
            # Create placeholder for the learning rate.
            learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
            # Compute the loss.
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.fcn8s_output), name='loss')
            # Compute the gradients and apply them.
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='adam_optimizer')
            train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')

        return loss, train_op, learning_rate, global_step

    def _build_predictor(self):

        with tf.name_scope('predictor'):

            softmax_output = tf.nn.softmax(self.fcn8s_output, name='softmax_output')

        return softmax_output

    def _build_metrics(self):

        with tf.variable_scope('metrics') as scope:

            # 1: Mean loss

            mean_loss_value, mean_loss_update_op = tf.metrics.mean(self.loss)

            mean_loss_value = tf.identity(mean_loss_value, name='mean_loss_value')
            mean_loss_update_op = tf.identity(mean_loss_update_op, name='mean_loss_update_op')

            # 1: Mean IoU

            mean_iou_value, mean_iou_update_op = tf.metrics.mean_iou(labels=self.labels,
                                                                     predictions=self.softmax_output,
                                                                     num_classes=self.num_classes)

            mean_iou_value = tf.identity(mean_iou_value, name='mean_iou_value')
            mean_iou_update_op = tf.identity(mean_iou_update_op, name='mean_iou_update_op')

            # 2: Accuracy

            acc_value, acc_update_op = tf.metrics.accuracy(labels=self.labels,
                                                           predictions=self.softmax_output)

            acc_value = tf.identity(acc_value, name='acc_value')
            acc_update_op = tf.identity(acc_update_op, name='acc_update_op')

            # TensorFlow's streaming metrics don't have reset operations,
            # so we need to create our own as a work-around. Say we want to evaluate
            # a metric after every training epoch. If we didn't have
            # a way to reset the metric's update op after every evaluation,
            # the computed metric value would be the average of the current evaluation
            # and all previous evaluations from past epochs, which is obviously not
            # what we want.
            local_metric_vars = tf.contrib.framework.get_variables(scope=scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            metrics_reset_op = tf.variables_initializer(var_list=local_metric_vars, name='metrics_reset_op')

        return (mean_loss_value,
                mean_loss_update_op,
                mean_iou_value,
                mean_iou_update_op,
                acc_value,
                acc_update_op,
                metrics_reset_op)

    def _build_summary_ops(self):

        graph = tf.get_default_graph()

        add_variable_summaries(variable=graph.get_tensor_by_name('pool3_1x1/kernel:0'), scope='pool3_1x1/kernel')
        add_variable_summaries(variable=graph.get_tensor_by_name('pool3_1x1/bias:0'), scope='pool3_1x1/bias')
        add_variable_summaries(variable=graph.get_tensor_by_name('pool4_1x1/kernel:0'), scope='pool4_1x1/kernel')
        add_variable_summaries(variable=graph.get_tensor_by_name('pool4_1x1/bias:0'), scope='pool4_1x1/bias')
        add_variable_summaries(variable=graph.get_tensor_by_name('fc7_1x1/kernel:0'), scope='fc7_1x1/kernel')
        add_variable_summaries(variable=graph.get_tensor_by_name('fc7_1x1/bias:0'), scope='fc7_1x1/bias')
        add_variable_summaries(variable=graph.get_tensor_by_name('fc7_conv2d_trans/kernel:0'), scope='fc7_conv2d_trans/kernel')
        add_variable_summaries(variable=graph.get_tensor_by_name('fc7_conv2d_trans/bias:0'), scope='fc7_conv2d_trans/bias')
        add_variable_summaries(variable=graph.get_tensor_by_name('fc7_pool4_conv2d_trans/kernel:0'), scope='fc7_pool4_conv2d_trans/kernel')
        add_variable_summaries(variable=graph.get_tensor_by_name('fc7_pool4_conv2d_trans/bias:0'), scope='fc7_pool4_conv2d_trans/bias')
        add_variable_summaries(variable=graph.get_tensor_by_name('fc7_pool4_pool3_conv2d_trans/kernel:0'), scope='fc7_pool4_pool3_conv2d_trans/kernel')
        add_variable_summaries(variable=graph.get_tensor_by_name('fc7_pool4_pool3_conv2d_trans/bias:0'), scope='fc7_pool4_pool3_conv2d_trans/bias')

        # Loss and learning rate.
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('learning_rate', self.learning_rate)

        summaries_training = tf.summary.merge_all()
        summaries_training = tf.identity(summaries_training, name='summaries_training')

        # All metrics.
        mean_loss = tf.summary.scalar('mean_loss', self.mean_loss_value)
        mean_iou = tf.summary.scalar('mean_iou', self.mean_iou_value)
        accuracy = tf.summary.scalar('accuracy', self.acc_value)

        summaries_evaluation = tf.summary.merge(inputs=[mean_loss,
                                                        mean_iou,
                                                        accuracy])
        summaries_evaluation = tf.identity(summaries_evaluation, name='summaries_evaluation')

        return summaries_training, summaries_evaluation

    def train(self,
              train_generator,
              epochs,
              steps_per_epoch,
              learning_rate_schedule,
              keep_prob=0.5,
              eval_dataset='train',
              eval_frequency=5,
              val_generator=None,
              val_steps=None,
              metrics={},
              save_during_training=False,
              save_dir=None,
              save_best_only=True,
              save_tags=['default'],
              save_frequency=5,
              monitor='loss',
              record_summaries=True,
              summaries_frequency=10,
              summaries_dir=None,
              summaries_name=None,
              training_loss_display_averaging=3):

        # Check for a GPU
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found. Please note that training this network will be unbearably slow without a GPU.')
        else:
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

        if not eval_dataset in ['train', 'val']:
            raise ValueError("`eval_dataset` must be one of 'train' or 'val', but is '{}'.".format(eval_dataset))

        if (eval_dataset == 'val') and ((val_generator is None) or (val_steps is None)):
            raise ValueError("When eval_dataset == 'val', a `val_generator` and `val_steps` must be passed.")

        for metric in metrics:
            if not metric in ['loss', 'mean_iou', 'accuracy']:
                raise ValueError("{} is not a valid metric. Valid metrics are ['loss', mean_iou', 'accuracy']".format(metric))

        if (not monitor in metrics) and (not monitor == 'loss'):
            raise ValueError('You are trying to monitor {}, but it is not in `metrics` and is therefore not being computed.'.format(monitor))

        self.eval_dataset = eval_dataset

        self.g_step = self.sess.run(self.global_step)
        learning_rate = learning_rate_schedule(self.g_step)

        # Set up the summary file writers.
        if record_summaries:
            training_writer = tf.summary.FileWriter(logdir=os.path.join(summaries_dir, summaries_name),
                                                    graph=self.sess.graph)
            if len(metrics) > 0:
                evaluation_writer = tf.summary.FileWriter(logdir=os.path.join(summaries_dir, summaries_name+'_eval'))

        for epoch in range(1, epochs+1):

            ##############################################################
            # Run the training for this epoch.
            ##############################################################

            loss_history = deque(maxlen=training_loss_display_averaging)

            tr = trange(steps_per_epoch, file=sys.stdout)
            tr.set_description('Epoch {}/{}'.format(epoch, epochs))

            for train_step in tr:

                batch_images, batch_labels = next(train_generator)

                if record_summaries and (self.g_step % summaries_frequency == 0):
                    _, current_loss, self.g_step, training_summary = self.sess.run([self.train_op,
                                                                                    self.loss,
                                                                                    self.global_step,
                                                                                    self.summaries_training],
                                                                                   feed_dict={self.image_input: batch_images,
                                                                                              self.labels: batch_labels,
                                                                                              self.learning_rate: learning_rate,
                                                                                              self.keep_prob: keep_prob})
                    training_writer.add_summary(summary=training_summary, global_step=self.g_step)
                else:
                    _, current_loss, self.g_step = self.sess.run([self.train_op,
                                                                  self.loss,
                                                                  self.global_step],
                                                                 feed_dict={self.image_input: batch_images,
                                                                            self.labels: batch_labels,
                                                                            self.learning_rate: learning_rate,
                                                                            self.keep_prob: keep_prob})

                self.variables_updated = True

                loss_history.append(current_loss)
                losses = np.array(loss_history)
                self.training_loss = np.mean(losses)

                tr.set_postfix(ordered_dict={'loss': self.training_loss,
                                             'learning rate': learning_rate})

                learning_rate = learning_rate_schedule(self.g_step)

            ##############################################################
            # Maybe evaluate the model after this epoch.
            ##############################################################

            if (len(metrics) > 0) and (epoch % eval_frequency == 0):

                if eval_dataset == 'train':
                    data_generator = train_generator
                    num_batches = steps_per_epoch
                    description = 'Evaluation on training dataset'
                elif eval_dataset == 'val':
                    data_generator = val_generator
                    num_batches = val_steps
                    description = 'Evaluation on validation dataset'

                self._evaluate(data_generator=data_generator,
                               metrics=metrics,
                               num_batches=num_batches,
                               description=description)

                if record_summaries:
                    evaluation_summary = self.sess.run(self.summaries_evaluation)
                    evaluation_writer.add_summary(summary=evaluation_summary, global_step=self.g_step)

            ##############################################################
            # Maybe save the model after this epoch.
            ##############################################################

            if save_during_training and (epoch % save_frequency == 0):

                save = False
                if save_best_only:
                    if (monitor == 'loss' and
                        (not 'loss' in self.metric_names) and
                        self.training_loss < self.best_training_loss):
                        save = True
                    else:
                        i = self.metric_names.index(monitor)
                        if self.metric_values[i] < self.best_metric_values[i]:
                            save = True
                    if save:
                        print('New best {} value, saving model.'.format(monitor))
                    else:
                        print('No improvement over previous best {} value, not saving model.'.format(monitor))
                else:
                    save = True

                if save:
                    self.save(model_save_dir=save_dir,
                              tags=save_tags,
                              name=None,
                              include_global_step=True,
                              include_last_training_loss=True,
                              include_metrics=(len(self.metric_names) > 0))

            ##############################################################
            # Update the current best metric values.
            ##############################################################

            if self.training_loss < self.best_training_loss:
                self.best_training_loss = self.training_loss

            if epoch % eval_frequency == 0:

                for i in range(len(self.metric_names)):
                    if self.metric_values[i] < self.best_metric_values[i]:
                        self.best_metric_values[i] = self.metric_values[i]

    def _evaluate(self, data_generator, metrics, num_batches, description='Running evaluation'):
        '''
        Internal method used by both `evaluate()` and `train()` that performs
        the actual evaluation. For the first three arguments, please refer
        to the documentation of the public `evaluate()` method.

        Arguments:
            description (string, optional): A description string that will be prepended
                to the progress bar while the evaluation is being processed. During
                training, this description is used to clarify whether the evaluation
                is being performed on the training or validation dataset.
        '''

        # Reset all metrics' accumulator variables.
        self.sess.run(self.metrics_reset_op)

        # Reset lists of previous tracked metrics.
        self.metric_names = []
        self.best_metric_values = []
        self.metric_update_ops = []
        self.metric_value_tensors = []

        # Set the metrics that will be evaluated.
        if 'loss' in metrics:
            self.metric_names.append('loss')
            self.best_metric_values.append(99999999.9)
            self.metric_update_ops.append(self.mean_loss_update_op)
            self.metric_value_tensors.append(self.mean_loss_value)
        if 'mean_iou' in metrics:
            self.metric_names.append('mean_iou')
            self.best_metric_values.append(0.0)
            self.metric_update_ops.append(self.mean_iou_update_op)
            self.metric_value_tensors.append(self.mean_iou_value)
        if 'accuracy' in metrics:
            self.metric_names.append('accuracy')
            self.best_metric_values.append(0.0)
            self.metric_update_ops.append(self.acc_update_op)
            self.metric_value_tensors.append(self.acc_value)

        # Set up the progress bar.
        tr = trange(num_batches, file=sys.stdout)
        tr.set_description(description)

        # Accumulate metrics in batches.
        for step in tr:

            batch_images, batch_labels = next(data_generator)

            self.sess.run(self.metric_update_ops,
                          feed_dict={self.image_input: batch_images,
                                     self.labels: batch_labels,
                                     self.keep_prob: 1.0})

        # Compute final metric values.
        self.metric_values = self.sess.run(self.metric_value_tensors)

        evaluation_results_string = ''
        for i, metric in enumerate(self.metric_names):
            evaluation_results_string += metric + ': {:.4f}  '.format(self.metric_values[i])
        print(evaluation_results_string)

    def evaluate(self, data_generator, metrics, num_batches):

        for metric in metrics:
            if not metric in ['loss', 'mean_iou', 'accuracy']:
                raise ValueError("{} is not a valid metric. Valid metrics are ['loss', mean_iou', 'accuracy']".format(metric))

        self._evaluate(data_generator, metrics, num_batches, description='Running evaluation')

    def predict(self, images):
        '''
        Makes predictions for the input images.

        Arguments:
            images (array-like): The input image or images. Must be an array-like
                object of rank 4. If predicting only one image, encapsulate it in
                a Python list.

        Returns:
            The prediction, an array of rank 4 of which the first three dimensions
            are identical to the input and the fourth dimension is the number of
            segmentation classes including the background class.
        '''

        return self.sess.run(self.softmax_output,
                             feed_dict={self.image_input: images,
                                        self.keep_prob: 1.0})

    def predict_and_save(self, results_dir, images_dir, image_size, color_map):
        '''

        Arguments:
            results_dir (string): The directory in which to save the annotated prediction
                output images. The images will be put inside a folder within this directory
                whose name is will be the current time stamp.
            images_dir (string): The directory in which the images to be processed are located.
            image_size (tuple): A tuple of the form `(image_height, image_width)` that
                represents the size to which all images will be resized.
            color_map (dictionary): A Python dictionary whose keys are non-negative
                integers representing segmentation classes and whose values are 1D tuples
                (or lists, Numpy arrays) of length 4 that represent the RGBA color values
                in which the respective classes are to be annotated. For example, if the
                dictionary contains the key-value pair `{1: (0, 255, 0, 127)}`, then
                this means that all pixels in the predicted image segmentation that belong
                to segmentation class 1 will be colored in green with 50% transparency
                in the input image.
        '''

        # Make a directory in which to store the results.
        output_dir = os.path.join(results_dir, str(time.time()))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        image_filepath_list = glob(os.path.join(images_dir, '*.png'))
        num_images = len(image_filepath_list)

        print('The segmented images will be saved to "{}"'.format(results_dir))

        tr = trange(num_images, file=sys.stdout)
        tr.set_description('Processing images')

        for i in tr:

            filepath = image_filepath_list[i]
            image = scipy.misc.imresize(scipy.misc.imread(filepath), image_size)

            prediction = self.sess.run(self.softmax_output,
                                       feed_dict={self.image_input: [image],
                                                  self.keep_prob: 1.0})

            output_image = print_segmentation_onto_image(image, prediction, color_map)

            scipy.misc.imsave(os.path.join(output_dir, os.path.basename(filepath)), output_image)

    def save(self,
             model_save_dir,
             tags=['default'],
             name=None,
             include_global_step=True,
             include_last_training_loss=True,
             include_metrics=True):

        if not self.variables_updated:
            print("Abort: Nothing to save, no training has been performed since the model was last saved.")
            return

        model_name = 'saved_model'
        if not name is None:
            model_name += '_' + name
        if include_global_step:
            model_name += '_(globalstep-{})'.format(self.g_step)
        if include_last_training_loss:
            model_name += '_(trainloss-{:.4f})'.format(self.training_loss)
        if include_metrics:
            if self.eval_dataset == 'val':
                model_name += '_(eval_on_val_dataset)'
            else:
                model_name += '_(eval_on_train_dataset)'
            for i in range(len(self.metric_names)):
                model_name += '_({}-{:.4f})'.format(self.metric_names[i], self.metric_values[i])
        if not (include_global_step or include_last_training_loss or include_metrics) and (name is None):
            model_name += '_{}'.format(time.time())

        saved_model_builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(model_save_dir, model_name))
        saved_model_builder.add_meta_graph_and_variables(sess=self.sess, tags=tags)
        saved_model_builder.save()

        self.variables_updated = False

    def close(self):
        self.sess.close()
        print("The session has been closed.")

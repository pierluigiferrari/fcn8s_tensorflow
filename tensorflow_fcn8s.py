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
from math import ceil
import time

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

        # When training, the latest metric evaluations will be stored here.
        # These values can be useful to include in the file names when saving the model.
        self.average_loss = None
        self.val_loss = None
        self.mean_iou = None
        self.accuracy = None
        self.g_step = None

        # Keep score of the best historical metric values.
        self.best_average_loss = 99999999.9
        self.best_val_loss = 99999999.9
        self.best_mean_iou = 0.0
        self.best_accuracy = 0.0

        self.sess = tf.Session()

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
            self.mean_iou_value = graph.get_tensor_by_name('evaluator/mean_iou_value:0')
            self.mean_iou_update_op = graph.get_tensor_by_name('evaluator/mean_iou_update_op:0')
            self.acc_value = graph.get_tensor_by_name('evaluator/acc_value:0')
            self.acc_update_op = graph.get_tensor_by_name('evaluator/acc_update_op:0')
            self.metrics_reset_op = graph.get_operation_by_name('evaluator/metrics_reset_op')

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
            self.mean_iou_value, self.mean_iou_update_op, self.acc_value, self.acc_update_op, self.metrics_reset_op = self._build_evaluator()
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

        # 1: Append 1x1 convolutions to the three output layers of the encoder to reduce the Number
        #    of channels to the number of classes.

        with tf.name_scope('decoder'):

            pool3_1x1 = tf.layers.conv2d(inputs=self.pool3_out,
                                         filters=self.num_classes,
                                         kernel_size=(1, 1),
                                         strides=(1, 1),
                                         padding='same',
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                         name='pool3_1x1')
            #pool3_elu = tf.nn.elu(pool3_1x1, name='pool3_elu')

            pool4_1x1 = tf.layers.conv2d(inputs=self.pool4_out,
                                         filters=self.num_classes,
                                         kernel_size=(1, 1),
                                         strides=(1, 1),
                                         padding='same',
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                         name='pool4_1x1')
            #pool4_elu = tf.nn.elu(pool4_1x1, name='pool4_elu')

            fc7_1x1 = tf.layers.conv2d(inputs=self.fc7_out,
                                       filters=self.num_classes,
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding='same',
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                       name='fc7_1x1')
            #fc7_elu = tf.nn.elu(fc7_1x1, name='fc7_elu')

            # 2: Upscale and fuse until we're back at the original image size.

            fc7_conv2d_trans = tf.layers.conv2d_transpose(inputs=fc7_1x1,
                                                          filters=self.num_classes,
                                                          kernel_size=(4, 4),
                                                          strides=(2, 2),
                                                          padding='same',
                                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                                          name='fc7_conv2d_trans')

            add_fc7_pool4 = tf.add(fc7_conv2d_trans, pool4_1x1, name='add_fc7_pool4')

            fc7_pool4_conv2d_trans = tf.layers.conv2d_transpose(inputs=add_fc7_pool4,
                                                                filters=self.num_classes,
                                                                kernel_size=(4, 4),
                                                                strides=(2, 2),
                                                                padding='same',
                                                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                                                name='fc7_pool4_conv2d_trans')

            add_fc7_pool4_pool3 = tf.add(fc7_pool4_conv2d_trans, pool3_1x1, name='add_fc7_pool4_pool3')

            fc7_pool4_pool3_conv2d_trans = tf.layers.conv2d_transpose(inputs=add_fc7_pool4_pool3,
                                                                      filters=self.num_classes,
                                                                      kernel_size=(16, 16),
                                                                      strides=(8, 8),
                                                                      padding='same',
                                                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
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

    def _build_evaluator(self):

        with tf.variable_scope('evaluator') as scope:

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

        return mean_iou_value, mean_iou_update_op, acc_value, acc_update_op, metrics_reset_op

    def train(self,
              train_generator,
              epochs,
              steps_per_epoch,
              learning_rate_schedule,
              keep_prob=0.5,
              val_generator=None,
              val_steps=None,
              compute_val_loss=True,
              evaluate=True,
              eval_dataset='train',
              eval_frequency=5,
              loss_display_averaging=3,
              save_during_training=False,
              save_dir=None,
              save_best_only=True,
              save_tags=['default'],
              save_frequency=5,
              monitor='loss'):

        # Check for a GPU
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found. Please note that training this network will be unbearably slow without a GPU.')
        else:
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

        self.g_step = self.sess.run(self.global_step)
        learning_rate = learning_rate_schedule(self.g_step)

        for epoch in range(1, epochs+1):

            ##############################################################
            # Run the training for this epoch.
            ##############################################################

            loss_history = deque(maxlen=loss_display_averaging)

            tr = trange(steps_per_epoch, file=sys.stdout)
            tr.set_description('Epoch {}/{}'.format(epoch, epochs))

            for train_step in tr:

                batch_images, batch_labels = next(train_generator)

                _, current_loss, self.g_step = self.sess.run([self.train_op, self.loss, self.global_step],
                                                             feed_dict={self.image_input: batch_images,
                                                                        self.labels: batch_labels,
                                                                        self.learning_rate: learning_rate,
                                                                        self.keep_prob: keep_prob})

                loss_history.append(current_loss)
                losses = np.array(loss_history)
                self.average_loss = np.mean(losses)

                tr.set_postfix(ordered_dict={'loss': self.average_loss,
                                             'learning rate': learning_rate})

                learning_rate = learning_rate_schedule(self.g_step)

            ##############################################################
            # Maybe evaluate the model after this epoch.
            ##############################################################

            if epoch % eval_frequency == 0:

                if compute_val_loss:

                    val_loss = 0
                    n_val_set = 0

                    tr = trange(val_steps, file=sys.stdout)
                    tr.set_description('Computing validation loss')

                    for val_step in tr:

                        batch_images, batch_labels = next(val_generator)
                        batch_val_loss = self.sess.run(self.loss,
                                                       feed_dict={self.image_input: batch_images,
                                                                  self.labels: batch_labels,
                                                                  self.keep_prob: 1.0})
                        val_loss += batch_val_loss * len(batch_images)
                        n_val_set += len(batch_images)

                    self.val_loss = val_loss / n_val_set

                    print('val_loss: {:.4f} '.format(self.val_loss))

                if evaluate:

                    if eval_dataset == 'train':

                        self._evaluate(data_generator=train_generator,
                                       num_batches=steps_per_epoch,
                                       description='Evaluation on training dataset')

                    elif eval_dataset == 'val':

                        self._evaluate(data_generator=val_generator,
                                       num_batches=val_steps,
                                       description='Evaluation on validation dataset')

            ##############################################################
            # Maybe save the model after this epoch.
            ##############################################################

            if save_during_training and (epoch % save_frequency == 0):

                save = False
                if save_best_only:
                    if monitor == 'loss' and (self.average_loss < self.best_average_loss):
                        save = True
                    if monitor == 'val_loss' and (self.val_loss < self.best_val_loss):
                        save = True
                    if monitor == 'mean_iou' and (self.mean_iou > self.best_mean_iou):
                        save = True
                    if monitor == 'accuracy' and (self.accuracy > self.best_accuracy):
                        save = True
                    if save:
                        print('New best {} value, saving model.'.format(monitor))
                else:
                    save = True

                if save:
                    self.save(model_save_dir=save_dir,
                              tags=save_tags,
                              include_global_step=True,
                              include_loss=True,
                              include_metrics=evaluate)

            ##############################################################
            # Update current best metric values.
            ##############################################################

            if self.average_loss < self.best_average_loss:
                self.best_average_loss = self.average_loss

            if epoch % eval_frequency == 0:

                if compute_val_loss and (self.val_loss < self.best_val_loss):
                    self.best_val_loss = self.val_loss

                if evaluate and (self.mean_iou > self.best_mean_iou):
                    self.best_mean_iou = self.mean_iou

                if evaluate and (self.accuracy > self.best_accuracy):
                    self.best_accuracy = self.accuracy

    def _evaluate(self, data_generator, num_batches, description):

        # Set up the progress bar.
        tr = trange(num_batches, file=sys.stdout)
        tr.set_description(description)

        # Accumulate metrics in batches.
        for step in tr:

            batch_images, batch_labels = next(data_generator)

            self.sess.run([self.mean_iou_update_op, self.acc_update_op],
                          feed_dict={self.image_input: batch_images,
                                     self.labels: batch_labels,
                                     self.keep_prob: 1.0})

        # Compute final metric values.
        self.mean_iou, self.accuracy = self.sess.run([self.mean_iou_value, self.acc_value])
        print('Accuracy: {:.4f}, Mean IoU: {:.4f}'.format(self.accuracy, self.mean_iou))

        # Reset all metrics' accumulator variables.
        self.sess.run(self.metrics_reset_op)

    def evaluate(self, data_generator, num_batches):

        self._evaluate(data_generator, num_batches, description='Running evaluation')

    def predict(self, images):

        return self.sess.run(self.softmax_output,
                             feed_dict={self.image_input: images,
                                        self.keep_prob: 1.0})

    def predict_and_save(self, results_dir, images_dir, image_size, annotation_map):
        '''

        Arguments:
            results_dir (string): The directory in which to save the annotated prediction
                output images. The images will be put inside a folder within this directory
                whose name is will be the current time stamp.
            images_dir (string): The directory in which the images to be processed are located.
            image_size (tuple): A tuple of the form `(image_height, image_width)` that
                represents the size to which all images will be resized.
            annonation_map (dictionary): A Python dictionary whose keys are non-negative
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

        print('Annotated images will be saved to "{}"'.format(results_dir))

        tr = trange(num_images, file=sys.stdout)
        tr.set_description('Processing images')

        for i in tr:

            filepath = image_filepath_list[i]
            image = scipy.misc.imresize(scipy.misc.imread(filepath), image_size)

            image_softmax = self.sess.run(self.softmax_output,
                                          feed_dict={self.image_input: [image],
                                                     self.keep_prob: 1.0})

            # Create a template of shape `(image_height, image_width, 4)` to store RGBA values.
            mask = np.zeros(shape=(image_size[0], image_size[1], 4), dtype=np.uint8)
            segmentation_map = np.squeeze(np.argmax(image_softmax, axis=-1))

            # Loop over all segmentation classes that are to be annotated and put their
            # color value at the respective image pixel.
            for segmentation_class, color_value in annotation_map.items():

                mask[segmentation_map == segmentation_class] = color_value

            mask = scipy.misc.toimage(mask, mode="RGBA")

            output_image = scipy.misc.toimage(image)
            output_image.paste(mask, box=None, mask=mask) # See http://effbot.org/imagingbook/image.htm#tag-Image.Image.paste for details.

            scipy.misc.imsave(os.path.join(output_dir, os.path.basename(filepath)), output_image)

    def save(self,
             model_save_dir,
             tags=['default'],
             include_global_step=True,
             include_loss=True,
             include_metrics=False):

        model_name = 'saved_model'
        if include_global_step:
            model_name += '_globalstep-{}'.format(self.g_step)
        if include_loss:
            if not self.val_loss is None:
                model_name += '_valloss-{:.4f}'.format(self.val_loss)
            else:
                model_name += '_loss-{:.4f}'.format(self.average_loss)
        if include_metrics:
            model_name += '_acc-{:.4f}_mIoU-{:.4f}'.format(self.accuracy, self.mean_iou)
        if not (include_global_step or include_loss or include_metrics):
            model_name += '_{}'.format(time.time())

        saved_model_builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(model_save_dir, model_name))
        saved_model_builder.add_meta_graph_and_variables(sess=self.sess, tags=tags)
        saved_model_builder.save()

    def close(self):
        self.sess.close()
        print("The session has been closed.")

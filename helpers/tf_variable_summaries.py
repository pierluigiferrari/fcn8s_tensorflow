import tensorflow as tf

def add_variable_summaries(variable, scope):
  '''
  Attach some summaries to a tensor for TensorBoard visualization, namely
  mean, standard deviation, minimum, maximum, and histogram.

  Arguments:
    var (TensorFlow Variable): A TensorFlow Variable of any shape to which to
        add summary operations. Must be a numerical data type.
  '''
  with tf.name_scope(scope):
    mean = tf.reduce_mean(variable)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(variable))
    tf.summary.scalar('min', tf.reduce_min(variable))
    tf.summary.histogram('histogram', variable)

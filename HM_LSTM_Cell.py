from tensorflow.python.layers import base as base_layer
from .utils import HM_LSTM_InputTuple, HM_LSTM_StateTuple
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops.rnn_cell_impl import _WEIGHTS_VARIABLE_NAME, _BIAS_VARIABLE_NAME, RNNCell
import tensorflow as tf

class HM_LSTM_Cell(RNNCell):

  def __init__(self,
               num_units,
               layer_id,
               slope_annealing_placeholder,
               forget_bias=1.0,
               reuse=False,
               name=None):

    super(HM_LSTM_Cell, self).__init__(_reuse=reuse, name=name)

    self._num_units = num_units
    self._layer_id = layer_id
    self._forget_bias = forget_bias
    self._slope = slope_annealing_placeholder

    self._initializer = None
    self._state_is_tuple = True

    self._state_size = HM_LSTM_StateTuple(num_units, num_units, 1)
    self._output_size = layer_id * num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def build(self, inputs_shape):
    h_depth = self._num_units

    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[3 * self._num_units, 4 * self._num_units + 1],
        initializer=self._initializer)

    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units + 1],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    sigmoid = math_ops.sigmoid
    tanh = math_ops.tanh
    hard_sigm = lambda x: tf.maximum(0, (tf.minimum(1, (self._slope * x + 1) / 2)))

    (h_t_below, z_t_below, h_prev_above) = inputs
    (c_prev, h_prev, z_prev) = state

    # f = forget_gate, i = input_gate, o = output_gate, g = new_input
    lstm_matrix = math_ops.matmul(
        array_ops.concat([
            h_prev,                # recurrent
            z_prev * h_prev_above, # top-down
            z_t_below * h_t_below  # bottom-up
        ], 1), self._kernel)
    lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

    ztilde_t = lstm_matrix[:,-1]
    lstm_matrix = lstm_matrix[:,0:-1]

    f, i, o, g = array_ops.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)

    f = sigmoid(f + self._forget_bias)
    i = sigmoid(i)
    o = sigmoid(o)
    g = tanh(g)
    ztilde_t = hard_sigm(ztilde_t)

    op_selector = tf.where(
        tf.equal(z_prev, tf.constant(0., dtype=tf.float32)),
        tf.where(
            tf.equal(z_t_below, tf.constant(1., dtype=tf.float32)),
            tf.constant('UPDATE', dtype=tf.string),
            tf.constant('COPY', dtype=tf.string)
        ),
        tf.constant('FLUSH', dtype=tf.string)
    )

    c = tf.where(
            tf.equal(op_selector, 'UPDATE'),
            f * c_prev + i * g,
            tf.where(
                tf.equal(op_selector, 'COPY'),
                c_prev,
                i * g
            )
    )

    h = tf.where(
        tf.equal(op_selector, 'COPY'),
        h_prev,
        o * tanh(c)
    )

    graph = tf.get_default_graph()
    with ops.name_scope('BinaryRound') as name:
        with graph.gradient_override_map({'Round': 'Identity'}):
            z = tf.round(ztilde_t, name=name)

    new_state = HM_LSTM_StateTuple(c, h, z)
    return h, new_state
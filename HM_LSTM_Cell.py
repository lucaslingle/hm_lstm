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
               slope_annealing_placeholder,
               forget_bias=1.0,
               reuse=False,
               name=None):

    super(HM_LSTM_Cell, self).__init__(_reuse=reuse, name=name)

    self._num_units = num_units
    self._forget_bias = forget_bias
    self._slope = slope_annealing_placeholder

    self._initializer = None
    self._state_is_tuple = True

    self._state_size = HM_LSTM_StateTuple(self._num_units, self._num_units, 1)
    self._output_size = self._num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def zero_state(self, batch_size, dtype):
      c = tf.zeros([batch_size, self._num_units])
      h = tf.zeros([batch_size, self._num_units])
      z = tf.zeros([batch_size, 1])
      return HM_LSTM_StateTuple(c=c, h=h, z=z)

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value  # Note: input to 'call' method is tf.concat([h_t_below, z_t_below, h_prev_above], 1)

    hb_dim = input_depth - 1 - self._output_size
    zb_dim = 1
    ha_dim = self._output_size

    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units + ha_dim + hb_dim, 4 * self._num_units + 1],
        initializer=self._initializer)

    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units + 1],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def input2tuple(self, inputs):
      # Note: input to 'call' method is tf.concat([h_t_below, z_t_below, h_prev_above], 1)
      input_depth = tf.shape(self._kernel)[1]

      hb_dim = input_depth - 1 - self._output_size
      zb_dim = 1
      ha_dim = self._output_size

      h_t_below = inputs[:, 0:hb_dim]
      z_t_below = inputs[:, hb_dim:(hb_dim + zb_dim)]
      h_prev_above = inputs[:, (hb_dim + zb_dim):(hb_dim + zb_dim + ha_dim)]

      return HM_LSTM_InputTuple(h_t_below, z_t_below, h_prev_above)

  def call(self, inputs, state):
    sigmoid = math_ops.sigmoid
    tanh = math_ops.tanh
    hard_sigm = lambda x: tf.maximum(0, (tf.minimum(1, (self._slope * x + 1) / 2)))

    (h_t_below, z_t_below, h_prev_above) = self.input2tuple(inputs)
    (c_prev, h_prev, z_prev) = state

    # f = forget_gate, i = input_gate, o = output_gate, g = new_input
    lstm_matrix = math_ops.matmul(
        array_ops.concat([
            h_prev,                # recurrent
            z_prev * h_prev_above, # top-down
            z_t_below * h_t_below  # bottom-up
        ], 1), self._kernel)
    lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

    ztilde_t = tf.expand_dims(lstm_matrix[:,-1], 1)
    lstm_matrix = lstm_matrix[:,0:-1]

    f, i, o, g = array_ops.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)

    f = sigmoid(f + self._forget_bias)
    i = sigmoid(i)
    o = sigmoid(o)
    g = tanh(g)
    ztilde_t = hard_sigm(ztilde_t)

    # We have to set multiple tensors using conditional computation based on the same condition.
    #
    # In the course of trying to do this, I discovered I had to tile some stuff.
    # See https://www.tensorflow.org/api_docs/python/tf/where

    batch_size = tf.shape(inputs)[0]

    UPDATE = tf.tile(input=tf.constant('UPDATE', dtype=tf.string), multiples=[batch_size])
    COPY = tf.tile(input=tf.constant('COPY', dtype=tf.string), multiples=[batch_size])
    FLUSH = tf.tile(input=tf.constant('FLUSH', dtype=tf.string), multiples=[batch_size])

    z_prev = tf.squeeze(z_prev, [1])
    z_t_below = tf.squeeze(z_t_below, [1])

    # this is a tensor of strings; each row is a string indicating which operation the cell ought to perform
    op_selector = tf.where(
        tf.equal(z_prev, tf.constant(0., dtype=tf.float32)),
        tf.where(
            tf.equal(z_t_below, tf.constant(1., dtype=tf.float32)),
            UPDATE,
            COPY
        ),
        FLUSH
    )

    # set the cell state based on which operation we're performing
    c = tf.where(
            tf.equal(op_selector, 'UPDATE'),
            f * c_prev + i * g,
            tf.where(
                tf.equal(op_selector, 'COPY'),
                c_prev,
                i * g
            )
    )

    # set the hidden state based on which operation we're performing
    h = tf.where(
        tf.equal(op_selector, 'COPY'),
        h_prev,
        o * tanh(c)
    )

    graph = tf.get_default_graph()
    with ops.name_scope('BinaryRound') as name:
        with graph.gradient_override_map({'Round': 'Identity'}):
            z = tf.round(ztilde_t, name=name)

    new_state = HM_LSTM_StateTuple(c=c, h=h, z=z)
    return h, new_state
from utils import HM_LSTM_InputTuple, HM_LSTM_StateTuple
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
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
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._forget_bias = forget_bias
    self._slope = slope_annealing_placeholder

    self._state_is_tuple = True

    self._kernel_initializer = tf.orthogonal_initializer()

    self._kernel_z_initializer = tf.glorot_uniform_initializer()

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
        shape=[self._num_units + ha_dim + hb_dim, 4 * self._num_units],
        initializer=self._kernel_initializer)

    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self._kernel_z =  self.add_variable(
        _WEIGHTS_VARIABLE_NAME + '_z',
        shape=[self._num_units + ha_dim + hb_dim, 1],
        initializer=self._kernel_z_initializer)

    self._bias_z = self.add_variable(
        _BIAS_VARIABLE_NAME + '_z',
        shape=[1],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def input2tuple(self, inputs):
      # Note: input to 'call' method is tf.concat([h_t_below, z_t_below, h_prev_above], 1)

      input_depth = tf.shape(inputs)[1]

      hb_dim = input_depth - 1 - self._output_size
      zb_dim = 1
      ha_dim = self._output_size

      h_t_below = inputs[:, 0:hb_dim]
      z_t_below = tf.expand_dims(inputs[:, hb_dim], 1)
      h_prev_above = inputs[:, (hb_dim + zb_dim):(hb_dim + zb_dim + ha_dim)]

      return HM_LSTM_InputTuple(h_t_below, z_t_below, h_prev_above)

  def call(self, inputs, state):
    sigmoid = math_ops.sigmoid
    tanh = math_ops.tanh

    input_ = self.input2tuple(inputs)

    (h_t_below, z_t_below, h_prev_above) = (input_.h_t_below, input_.z_t_below, input_.h_prev_above)
    (c_prev, h_prev, z_prev) = (state.c, state.h, state.z)

    # f = forget_gate, i = input_gate, o = output_gate, g = new_input
    lstm_matrix = math_ops.matmul(
        array_ops.concat([
            h_prev,                # recurrent
            z_prev * h_prev_above, # top-down
            z_t_below * h_t_below  # bottom-up
        ], 1), self._kernel)
    lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

    ztilde = math_ops.matmul(
        array_ops.concat([
            h_prev,                # recurrent
            z_prev * h_prev_above, # top-down
            z_t_below * h_t_below  # bottom-up
        ], 1), self._kernel_z)
    ztilde = nn_ops.bias_add(ztilde, self._bias_z)

    f, i, o, g = array_ops.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)

    f = sigmoid(f + self._forget_bias)
    i = sigmoid(i)
    o = sigmoid(o)
    g = tanh(g)

    ztilde_t = tf.maximum(0.0, tf.minimum(1.0, ((self._slope * ztilde + 1.0) / 2.0)))

    # set the cell state based on which operation we're performing
    c = tf.where(
        tf.equal(tf.squeeze(z_prev, [1]), tf.constant(0., dtype=tf.float32)),
        tf.where(
            tf.equal(tf.squeeze(z_t_below, [1]), tf.constant(1., dtype=tf.float32)),
            f * c_prev + i * g,                    # UPDATE
            c_prev                                 # COPY
        ),
        i * g                                      # FLUSH
    )

    # set the hidden state based on which operation we're performing
    h = tf.where(
        tf.logical_and(
            tf.equal(tf.squeeze(z_t_below, [1]), tf.constant(0., dtype=tf.float32)),
            tf.equal(tf.squeeze(z_prev, [1]), tf.constant(0., dtype=tf.float32))),
        h_prev,
        o * tanh(c)
    )

    graph = tf.get_default_graph()
    with ops.name_scope("ST_Round") as name:
        with graph.gradient_override_map({"Round": "Identity"}):
            z = tf.round(ztilde_t, name=name)

    new_state = HM_LSTM_StateTuple(c=c, h=h, z=z)
    return h, new_state
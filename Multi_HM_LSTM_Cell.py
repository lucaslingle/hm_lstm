from utils import HM_LSTM_InputTuple, HM_LSTM_StateTuple
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import ops
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell
import tensorflow as tf

class Multi_HM_LSTM_Cell(MultiRNNCell):

  def __init__(self, cells, output_embedder=None):
    super(Multi_HM_LSTM_Cell, self).__init__(cells=cells)
    self._state_is_tuple = True

    self._output_embedder = output_embedder

  @property
  def state_size(self):
    return tuple(cell.state_size for cell in self._cells)

  @property
  def output_size(self):
    if self._output_embedder is None:
      return tuple(cell.output_size for cell in self._cells)
    else:
      return self._output_embedder.output_size


  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)

  def call(self, inputs, state):
    """Run this multi-layer cell on inputs, starting from state."""
    cur_h_t_below = inputs
    cur_z_t_below = tf.ones(dtype=tf.float32, shape=[tf.shape(inputs)[0], 1])

    new_states = []
    layer_outputs = []

    for i, cell in enumerate(self._cells):
      with vs.variable_scope("cell_%d" % i):

        if not nest.is_sequence(state):
          raise ValueError("Expected state to be a tuple of length %d, but received: %s" % (len(self.state_size), state))

        cur_state = state[i]

        if i == len(self._cells) - 1:
          h_prev_above = tf.zeros(dtype=tf.float32, shape=[tf.shape(inputs)[0], self._cells[i].output_size])
        else:
          h_prev_above = tf.identity(state[i + 1].h)

        cur_inp = tf.concat([cur_h_t_below, cur_z_t_below, h_prev_above], 1)
        h, new_state = cell(cur_inp, cur_state)

        if i == len(self._cells) - 1:
          new_c, new_h, new_z = (new_state.c, new_state.h, new_state.z)
          new_z = tf.zeros_like(new_z, dtype=tf.float32)
          new_state = HM_LSTM_StateTuple(c=new_c, h=new_h, z=new_z)

        new_states.append(new_state)
        layer_outputs.append(h)

        cur_h_t_below = tf.identity(h)
        cur_z_t_below = tf.identity(new_state.z)

    new_states = tuple(new_states)
    layer_outputs = tuple(layer_outputs)

    if self._output_embedder is not None:
      h_out = self._output_embedder.apply(layer_outputs)
    else:
      h_out = layer_outputs

    return h_out, new_states
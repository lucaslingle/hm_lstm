import collections
import tensorflow as tf

_HM_LSTM_InputTuple = collections.namedtuple("HM_LSTM_InputTuple", ("h_t_below", "z_t_below", "h_prev_above"))

class HM_LSTM_InputTuple(_HM_LSTM_InputTuple):
  """Tuple used by HM LSTM Cells to organize inputs.
  Stores three elements: `(h_t_below, z_t_below, h_prev_above)`, in that order.
  Where `h_t_below` is the hidden state at time t of layer below,
  and `z_t_below` is the boundary detector value at time t of the layer below,
  and `h_prev_above` is the hidden state at time
  """
  __slots__ = ()

  @property
  def dtype(self):
    (h_t_below, z_t_below, h_prev_above) = self
    if h_t_below.dtype != z_t_below.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(h_t_below.dtype), str(z_t_below.dtype)))
    elif h_t_below.dtype != h_prev_above.dtype:
        raise TypeError("Inconsistent internal state: %s vs %s" %
                        (str(h_t_below.dtype), str(h_prev_above.dtype)))
    return h_t_below.dtype



_HM_LSTM_StateTuple = collections.namedtuple("HM_LSTM_StateTuple", ("c", "h", "z"))

class HM_LSTM_StateTuple(_HM_LSTM_StateTuple):
  """Tuple used by HM LSTM Cells to organize cell state.
  Stores three elements: `(c, h, z)`, in that order.
  Where `c` is the cell state, `h` is the hidden state, and `z` is the boundary detector for the cell.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h, z) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    elif c.dtype != z.dtype:
        raise TypeError("Inconsistent internal state: %s vs %s" %
                        (str(c.dtype), str(z.dtype)))
    return c.dtype


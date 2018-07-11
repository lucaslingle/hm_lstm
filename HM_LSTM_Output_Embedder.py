import tensorflow as tf

class HM_LSTM_Output_Embedder:

  def __init__(self, num_layers, hidden_dim, output_emb_dim, activation=tf.nn.relu):

      self._num_layers = num_layers
      self._hidden_dim = hidden_dim
      self._output_emb_dim = output_emb_dim
      self._activation = activation

  @property
  def output_size(self):
      return self._output_emb_dim

  def apply(self, layer_outputs):
      assert type(layer_outputs) == tuple
      assert len(layer_outputs) == self._num_layers
      assert layer_outputs[0].get_shape().as_list()[-1] == self._hidden_dim

      # layer_outputs is a tuple of length num_layers of the layers' outputs.
      # each layer output has shape [batch size, hidden_dim]

      h_out = tf.concat([
          tf.expand_dims(layer_outputs[idx], 1) for idx in range(0,len(layer_outputs))
      ], 1)
      # shape: [batch_size, num_layers, hidden_dim]

      h_out_flat = tf.concat([
          layer_outputs[idx] for idx in range(0,len(layer_outputs))
      ], 1)
      # shape: [batch_size, num_layers * hidden_dim]

      layer_importance_gates = tf.layers.dense(h_out_flat, units=self._num_layers, use_bias=False, activation=tf.sigmoid)
      layer_importance_gates = tf.expand_dims(layer_importance_gates, 2)

      h_out_importance_gated = layer_importance_gates * h_out
      # shape: [batch_size, num_layers, 1] * [batch_size, num_layers, hidden_dim] ->  # [batch_size, num_layers, hidden_dim]

      h_out_importance_gated_flat = tf.reshape(
          h_out_importance_gated, [-1, self._num_layers * self._hidden_dim])

      h_out_emb = tf.layers.dense(h_out_importance_gated_flat, units=self._output_emb_dim, use_bias=False, activation=self._activation)
      # shape: [batch_size, output_emb_dim]

      return h_out_emb




Hierarchical Multiscale Recurrent Neural Networks in Tensorflow
========================================

Implementation of Chung et al., 2016 - 'Hierarchical Multiscale Recurrent Neural Networks'

![alt tag](assets/hmrnn_picture.png?raw=true)

How to use:
-----------

if you clone this repo to a directory `/path/to/cloned/repo/`, you can create a Hierarchical Multisale LSTM as follows:

```
import sys

hm_lstm_module_dir = '/path/to/cloned/repo/'

if hm_lstm_module_dir not in sys.path:
    sys.path.append(hm_lstm_module_dir)

from HM_LSTM_Cell import HM_LSTM_Cell
from Multi_HM_LSTM_Cell import Multi_HM_LSTM_Cell
from HM_LSTM_Output_Embedder import HM_LSTM_Output_Embedder
from utils import HM_LSTM_StateTuple

slope_value = tf.placeholder(dtype=tf.float32, shape=[])

hm_lstm_cells = [
    HM_LSTM_Cell(
        num_units=params.hidden_dim, 
        slope_annealing_placeholder=slope_value,
        forget_bias=1.0)
    for _ in range(0, params.num_layers)
]
    
output_embedder = HM_LSTM_Output_Embedder(
    num_layers=params.num_layers, 
    hidden_dim=params.hidden_dim, 
    output_emb_dim=params.output_emb_dim,
    activation=tf.nn.relu)

multi_hm_lstm_cell = Multi_HM_LSTM_Cell(
    cells=hm_lstm_cells, 
    output_embedder=output_embedder)
```

and proceed to use `multi_hm_lstm_cell` like a `MultiRNNCell`, e.g., use it with `tf.nn.dynamic_rnn` or provide it to `tf.contrib.seq2seq.BasicDecoder`.
Over the course of training, the value provided to `slope_value` placeholder should be annealed linearly from a small value like 0.5 to a large value like 5.0.

For an extensive example of how to use this module, please refer to [this jupyter notebook](https://github.com/lucaslingle/estimators/blob/master/imdb_reviews_language_model/imdb_language_model_hmlstm.ipynb)
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow_hub as hub
import tensorflow as tf

class OneHotLayer(layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        

    def build(self, input_shape):
        super(OneHotLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return tf.one_hot(indices=tf.cast(x, dtype=tf.int32), depth=self.input_dim)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.input_dim)
    
    def get_config(self):
        config = {
            'input_dim': self.input_dim,
        }
        base_config = super(OneHotLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    
class ElmoLayer(layers.Layer):
    def __init__(self, trainable=True, output_type="default", signature="default", output_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = 1024
        self.signature = signature
        self.output_type = output_type
        self.trainable=trainable
        
    def build(self, input_shape):
        self.elmo = hub.Module(
            'https://tfhub.dev/google/elmo/2',
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        if self.trainable:
            self._trainable_weights.extend(tf.trainable_variables(scope="^{}_module/.*".format(self.name)))
            
        super(ElmoLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(
            tf.squeeze(tf.cast(x, dtype=tf.string), axis=1),
            as_dict=True,
            signature=self.signature
        )[self.output_type]
        return result

    def compute_output_shape(self, input_shape):
        if self.output_type == "default":
            return (input_shape[0], self.output_dim)
        else:
            return (input_shape[0], None, self.output_dim)
        
    def get_config(self):
         config = {
            "output_size": self.output_size,
            "signature": self.signature,
            "output_type": self.output_type,
            "trainable": self.trainable,
        }
    
    
    
# dict_output
# Use "pooled_output" for classification tasks on an entire sentence.
# Use "sequence_output" for token-level output.
# signature
# Use "tokens", because only that option is currently supported(04.2019), assumes pre-processed inputs
class BertLayer(layers.Layer):
    def __init__(self, trainable=True, output_type="pooled_output", signature="tokens", output_dim=768, n_fine_tune_layers=3, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = 768
        self.n_fine_tune_layers = n_fine_tune_layers
        self.signature = signature
        self.output_type = output_type
        self.trainable=trainable
        

    def build(self, input_shape):
        self.bert = hub.Module(
            "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        if self.trainable:
            trainable_vars = self.bert.variables

            # Remove unused layers
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name and not "/pooler/" in var.name]

            # Select how many layers to fine tune
            trainable_vars = trainable_vars[-self.n_fine_tune_layers :]

            # Add to trainable weights
            for var in trainable_vars:
                self._trainable_weights.append(var)

            # Add non-trainable weights
            for var in self.bert.variables:
                if var not in self._trainable_weights:
                    self._non_trainable_weights.append(var)
        
        super().build(input_shape)

    def call(self, inputs):
        inputs = [tf.cast(x, dtype=tf.int32) for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids
        )
        result = self.bert(
            inputs=bert_inputs,
            as_dict=True,
            signature=self.signature
        )[self.output_type]
        return result

    def compute_output_shape(self, input_shape):
        if self.output_type == "pooled_output":
            return (input_shape[0], self.output_dim)
        else:
            return (input_shape[0], None, self.output_dim)

class USELayer(layers.Layer):
    def __init__(self, trainable=True, output_dim=512, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = 512
        self.trainable=trainable
        
    def build(self, input_shape):
        self.use = hub.Module(
            'https://tfhub.dev/google/universal-sentence-encoder-large/3',
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        if self.trainable:
            self._trainable_weights.extend(tf.trainable_variables(scope="^{}_module/.*".format(self.name)))
            
        super().build(input_shape)

    def call(self, x, mask=None):
        result = self.use(tf.squeeze(tf.cast(x, tf.string)))
        return result
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
        
    def get_config(self):
         config = {
            "output_size": self.output_size,
            "trainable": self.trainable,
        }
        
        
class BertInputLayer():
    def __init__(self, shape, dtype):
        self.input_id = layers.Input(shape=shape, dtype=dtype, name="input_id")
        self.input_mask = layers.Input(shape=shape, dtype=dtype, name="input_mask")
        self.input_segment = layers.Input(shape=shape, dtype=dtype, name="input_segment")

    def forward(self):
        return [self.input_id, self.input_mask, self.input_segment]
    
    
    
class AttensionLayer(layers.Layer):
    def __init__(self, trainable=True):
        pass
    

    
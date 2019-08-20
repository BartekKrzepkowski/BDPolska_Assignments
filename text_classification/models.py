import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from layers import BertInputLayer, BertLayer, ElmoLayer, USELayer, OneHotLayer


embeddings = {
    "Bert": BertLayer,
    "Elmo": ElmoLayer,
    "USE": USELayer,
    "Embedding": layers.Embedding,
    "OneHot": OneHotLayer
}

input_ = {
    "simple_input": layers.Input,
    "bert_input": lambda shape, dtype: BertInputLayer(shape, dtype).forward()
}


def model_rnn(kwargs):
    K.clear_session()
    nn_input = input_[kwargs["input_layer"]](**kwargs["input_params"])
    x = embeddings[kwargs["emb_layer"]](**kwargs["emb_params"])(nn_input)
    
    x = layers.SpatialDropout1D(0.1)(x)
    x = layers.Bidirectional(layers.GRU(32, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(16, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.concatenate([avg_pool, max_pool])
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(100)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    
    nn_pred = layers.Dense(kwargs["out_units"], activation=kwargs["out_activation"])(x)
    
    model = Model(inputs=nn_input, outputs=nn_pred)
    
    model.compile(
        loss=kwargs["loss"],
        optimizer=kwargs["optimizer"],
        metrics=["accuracy"]
    )
    sess = K.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    return model


# Architektura Yoon Kima z dynamicznym wejściem
def model_cnn(kwargs):
    K.clear_session()
    nn_input = input_[kwargs["input_layer"]](**kwargs["input_params"])
    x = embeddings[kwargs["emb_layer"]](**kwargs["emb_params"])(nn_input)
    
    x = layers.SpatialDropout1D(0.1)(x)
    x = layers.Reshape((-1, kwargs["emb_params"]["output_dim"], 1), input_shape=K.int_shape(x))(x)
    
    maxpool_pool = []
    filter_sizes = [1, 2, 3, 5]
    for i in range(len(filter_sizes)):
        conv = layers.Conv2D(32, kernel_size=(filter_sizes[i], kwargs["emb_params"]["output_dim"]),
                                     kernel_initializer='he_normal', activation='relu')(x)
        conv = layers.Reshape((-1, K.int_shape(conv)[3]))(conv)
        global_pool = layers.GlobalAveragePooling1D()(conv)
        maxpool_pool.append(global_pool)
        
    x = layers.Concatenate(axis=1)(maxpool_pool)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(32)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    
    nn_pred = layers.Dense(kwargs["out_units"], activation=kwargs["out_activation"])(x)
    
    model = Model(inputs=nn_input, outputs=nn_pred)
    
    model.compile(
        loss=kwargs["loss"],
        optimizer=kwargs["optimizer"],
        metrics=["accuracy"]
    )
    sess = K.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    return model


def model_ffnn(kwargs):
    K.clear_session()   
    nn_input = input_[kwargs["input_layer"]](**kwargs["input_params"])
    x = embeddings[kwargs["emb_layer"]](**kwargs["emb_params"])(nn_input)
       
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)
    
    nn_pred = layers.Dense(kwargs["out_units"], activation=kwargs["out_activation"])(x)
    
    model = Model(inputs=nn_input, outputs=nn_pred)
    
    model.compile(
        loss=kwargs["loss"],
        optimizer=kwargs["optimizer"],
        metrics=["accuracy"]
    )
    sess = K.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.tables_initializer())
    
    return model






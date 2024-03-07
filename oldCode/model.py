from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.initializers import HeNormal
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras.optimizers.schedules import ExponentialDecay
from keras.layers import Dropout
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.losses import CategoricalCrossentropy 
from keras.layers import MultiHeadAttention, Input, Reshape
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


class NLTK_Classifier:
    def __init__(self, numClasses=2, inputShape=10):
        inputs = Input(shape=(inputShape,))
        inputs = Reshape((1, inputShape))(inputs)
        transformer_output = self.transformLayer(heads=3)(inputs, inputs, inputs)
        dense_output = Dense(16, activation='relu')(transformer_output)
        outputs = Dense(numClasses, activation='softmax')(dense_output)
        outputs = Reshape((numClasses,))(outputs)
        
        self.model = Model(inputs=inputs, outputs=outputs)

    # custom transformer layer 
    """
    A transformer layer expects inputs to be of shape (batch_size, seq_len, dimension of each input vector)
    """
    def transformLayer(self, heads):
        return MultiHeadAttention(
                num_heads=heads, key_dim=8, dropout=0.1,
                kernel_initializer='he_normal', bias_initializer='zeros',
                bias_regularizer=l2(0.01), kernel_regularizer=l2(0.01)
        )
    
    # Customer Dense layer
    def DenseLayer(self, nodes, activation='relu'):
        return Dense(
            nodes, activation=activation, 
            kernel_initializer=HeNormal(), bias_initializer=HeNormal(),
            kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)
        )

    # Resets weights to HeNormal
    def reset_weights(self):
        initial_weights = self.model.get_weights()
        self.model.set_weights(initial_weights)

    # compile the model
    def compile(self):
        self.model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

    # Run the model. Forward fit using a learning rate scheduler
    def fit(self, X_train, training_labels, epochs=1, batch_size=32):
        lr_scheduler = ExponentialDecay(initial_learning_rate=0.001, decay_steps=5, decay_rate=0.5)
        self.model.fit(X_train, training_labels, epochs=epochs, 
                    batch_size=batch_size, callbacks=[LearningRateScheduler(lr_scheduler)])

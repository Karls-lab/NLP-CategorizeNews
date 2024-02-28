from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.initializers import HeNormal
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras.optimizers.schedules import ExponentialDecay
from keras.layers import Dropout
from sklearn.base import BaseEstimator, ClassifierMixin


class NLTK_Binary_Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = Sequential([
            self.DenseLayer(32, activation='relu'),
            Dropout(0.2),
            self.DenseLayer(1, activation='sigmoid'),
        ])

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
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Run the model. Forward fit using a learning rate scheduler
    def fit(self, training_images, training_labels, epochs=1, batch_size=32):
        lr_scheduler = ExponentialDecay(initial_learning_rate=0.001, decay_steps=1, decay_rate=.1)
        self.model.fit(training_images, training_labels, epochs=epochs, 
                    batch_size=batch_size, callbacks=[LearningRateScheduler(lr_scheduler)])

# display model summary
model = NLTK_Binary_Classifier()
model.model.summary()

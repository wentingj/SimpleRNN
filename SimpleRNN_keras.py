from keras.layers import Input
from keras.models import Model
from keras.layers import LSTM, SimpleRNN
import numpy as np
import keras.backend as K

data_dim = 16
timesteps = 8
units = 10
batch_size = 10

#check backward
input_ = Input(shape=(timesteps, data_dim))
RNN = SimpleRNN(units, unroll=True, implementation=1)(input_)

model = Model(input=input_, output=RNN)
model.compile(optimizer='sgd', loss='categorical_crossentropy')

weight = model.get_weights()

weights_list = model.trainable_weights # weight tensors
gradients = model.optimizer.get_gradients(model.total_loss, weights_list) # gradient tensors

input_tensors = [model.inputs[0], # input data
                 model.sample_weights[0], # how much to weight each sample by
                 model.targets[0], # labels
                 K.learning_phase(), # train or test mode
]

get_gradients = K.function(inputs=input_tensors, outputs=gradients)

inputs = [np.random.randn(batch_size, timesteps, data_dim), # X
          np.ones(batch_size), # sample weights
          np.random.randint(2, size=[batch_size, units]), # y
          0 # learning phase in TEST mode
]
print "weights_list:"
print weights_list
print "weight:"
print weight
print "gradients:"
print zip(weights_list, get_gradients(inputs))

################################################################################
#check forward
get_output = K.function([model.inputs[0]], [model.output])
print "outputs:"
print get_output([inputs[0]])

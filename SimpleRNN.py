import theano
from theano import tensor as T
import numpy as np
import sys
import mkl_simplernn_op
from mkl_simplernn_op import SimpleRNN

units = 1000#int(sys.argv[1])
timesteps = 32#int(sys.argv[2])
batch_size = 80#int(sys.argv[3])
input_dim = 620#int(sys.argv[4])

print "units=",units
print "timesteps=",timesteps
print "batch_size=",batch_size
print "input_dim=",input_dim

x = np.random.rand(timesteps, input_dim, batch_size).astype(np.float64)
h_init = np.random.rand(units, batch_size).astype(np.float64)-np.random.rand(units, batch_size).astype(np.float64)
w_x = np.random.rand(units, input_dim).astype(np.float64)-np.random.rand(units, input_dim).astype(np.float64)
w_h = np.random.rand(units, units).astype(np.float64)-np.random.rand(units, units).astype(np.float64)
b = np.zeros((timesteps, units, batch_size), dtype=np.float64)

def SimpleRNN_NP():
    global x, h_init, w_x, w_h, b
    h_t = h_init
    for i in range(timesteps):
	h_t = np.tanh(np.dot(w_x, x[i]) + np.dot(w_h, h_t) + b[i])
    return h_t
   
def SimpleRNN_MKL():
    global x, h_init, w_x, w_h, b
    X = T.dtensor3('X')
    H_init = T.dmatrix('H_init')
    W_x = T.dmatrix('W_x')
    W_h = T.dmatrix('W_h')
    B = T.dtensor3('B') 
    o = SimpleRNN()(X, H_init, W_x, W_h, B)
    f = theano.function([X, H_init, W_x, W_h, B], o)

    o_mkl = f(x, h_init, w_x, w_h, b)
    return o_mkl

if __name__ == '__main__':
    o_numpy = SimpleRNN_NP()
    o_mkl = SimpleRNN_MKL()
    print o_numpy
    print o_mkl
    print (o_numpy-o_mkl)/o_numpy
    #assert np.allclose(o_numpy, o_mkl)

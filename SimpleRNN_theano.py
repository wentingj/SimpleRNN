import time
import theano
import numpy as np

from theano import tensor as T
from theano.tensor import tanh
import mkl_simplernn_bw_op
from mkl_simplernn_bw_op import SimpleRNN_bw 

X = T.ftensor3('X')
W_x = T.fmatrix('W_x')
W_h = T.fmatrix('W_h')
B = T.fvector('B')
B_mkl = T.fmatrix('B_mkl')
hid = T.fmatrix('hid')
o_real = T.ftensor3('o_real') 

def step(x, h_tm1):
    global W_h, B
    h_t = tanh(x + T.dot(h_tm1, W_h) + B)
    return h_t

def SimpleRNN_theano():
    global X, W_x, hid
    X_r = T.dot(X, W_x)
    fn = lambda x_r, h_tm1: step(x_r, h_tm1)
    result, updates = theano.scan(fn, sequences=[ X_r], outputs_info=hid, name='test_theano_gru_scan')
    return result


if __name__ == '__main__':
    input_dim = 2
    timesteps = 3
    units = 3
    batch_size = 2
    print "input_dim",input_dim
    print "timesteps",timesteps
    print "units",units
    print "batch_size",batch_size

    o = SimpleRNN_theano()

    v_x = np.random.rand(timesteps, batch_size, input_dim).astype(np.float32)
    v_w_x = np.random.rand(input_dim, units).astype(np.float32) - np.random.rand(input_dim, units).astype(np.float32)
    v_w_h = np.random.rand(units, units).astype(np.float32) - np.random.rand(units, units).astype(np.float32)

    v_b = np.random.rand(units).astype(np.float32) - np.random.rand(units).astype(np.float32)
    v_b_mkl = np.zeros((batch_size, units), dtype=np.float32)
    for i in range(batch_size):
        v_b_mkl[i] = v_b

    v_hid = np.zeros((batch_size, units), np.float32)
    v_o_real = np.random.rand(timesteps, batch_size, units).astype(np.float32)

    #theano#################################################################################################################
    #check forward
    fo = theano.function([X, W_x, W_h, B, hid], o)
    print"x=", v_x
    print"w_x=",v_w_x
    print"w_h=", v_w_h
    print"b=", v_b
    print"b_mkl=", v_b_mkl
    print"hid=",v_hid
    print"o_real=",v_o_real

    v_o = fo(v_x, v_w_x, v_w_h, v_b, v_hid)
    #print "forward o=",v_o

    #check gradients
    loss = -T.sum(o * T.log(o_real))
    gx = T.grad(loss, X)
    fx = theano.function([X, W_x, W_h, B, hid, o_real], gx)
    #theano.printing.pydotprint(fx, outfile='rnn_dx.png', var_with_name_simple=True)
    gradients_x = fx(v_x, v_w_x, v_w_h, v_b, v_hid, v_o_real)
    print "gradients_x=", gradients_x

    #mkl#################################################################################################################
    O = mkl_simplernn_bw_op.SimpleRNN_bw()(X, W_x, W_h, B_mkl, hid, o_real)
    fx_mkl = theano.function([X, W_x, W_h, B_mkl, hid, o_real], O)
    gradients_x_mkl = fx_mkl(v_x, v_w_x, v_w_h, v_b_mkl, v_hid, v_o_real)
    theano.printing.pydotprint(fx_mkl, outfile='rnn_dx.png', var_with_name_simple=True)
    
    print "gradients_x_mkl=", gradients_x_mkl

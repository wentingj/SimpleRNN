import time
import theano
import numpy as np

from theano import tensor as T
from theano.tensor import tanh

X = T.ftensor3('X')
W_x = T.fmatrix('W_x')
W_h = T.fmatrix('W_h')
B = T.fvector('B')
hid = T.fmatrix('hid')
o_real = T.fmatrix('o_real') 

def step(x, h_tm1):
    global W_h, B
    h_t = tanh(x + T.dot(h_tm1, W_h) + B)
    return h_t

def GRU_theano():
    global X, W_x, hid
    X_r = T.dot(X, W_x)
    fn = lambda x_r, h_tm1: step(x_r, h_tm1)
    result, updates = theano.scan(fn, sequences=[ X_r], outputs_info=hid, name='test_theano_gru_scan')
    return result

if __name__ == '__main__':
    input_dim = 16
    timesteps = 8
    units = 10
    batch_size = 10
    print "input_dim",input_dim
    print "timesteps",timesteps
    print "units",units
    print "batch_size",batch_size

    o = GRU_theano()

    v_x = np.random.rand(timesteps, batch_size, input_dim).astype(np.float32)
    v_w_x = np.random.rand(input_dim, units).astype(np.float32) - np.random.rand(input_dim, units).astype(np.float32)
    v_w_h = np.random.rand(units, units).astype(np.float32) - np.random.rand(units, units).astype(np.float32)

    v_b = np.random.rand(units).astype(np.float32) - np.random.rand(units).astype(np.float32)

    v_hid = np.zeros((batch_size, units), np.float32)
    v_o_real = np.random.rand(batch_size, units).astype(np.float32)

    #check forward
    fo = theano.function([X, W_x, W_h, B, hid], o)
    v_o = fo(v_x, v_w_x, v_w_h, v_b, v_hid)
    print "forward o=",v_o

    #check gradients
    loss = -T.sum(o * T.log(o_real))
    gx = T.grad(loss, X)
    gwx = T.grad(loss, W_x)
    gwh = T.grad(loss, W_h)
    gb = T.grad(loss, B)
    
    fx = theano.function([X, W_x, W_h, B, hid, o_real], gx)
    fwx = theano.function([X, W_x, W_h, B, hid, o_real], gwx)
    fwh = theano.function([X, W_x, W_h, B, hid, o_real], gwh)
    fb = theano.function([X, W_x, W_h, B, hid, o_real], gb)

    gradients_x = fx(v_x, v_w_x, v_w_h, v_b, v_hid, v_o_real)
    gradients_wx = fwx(v_x, v_w_x, v_w_h, v_b, v_hid, v_o_real)
    gradients_wh = fwh(v_x, v_w_x, v_w_h, v_b, v_hid, v_o_real)
    gradients_b = fb(v_x, v_w_x, v_w_h, v_b, v_hid, v_o_real)

    print "gradients_x=", gradients_x
    print "gradients_wx=", gradients_wx
    print "gradients_wh=", gradients_wh
    print "gradients_b=", gradients_b

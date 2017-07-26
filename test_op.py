import theano
from theano import tensor as T
import numpy as np
import sys
import mkl_simplernn_op
from mkl_simplernn_op import SimpleRNN

np.random.seed(12345)

units = 1000    # int(sys.argv[1])
timesteps = 4   # int(sys.argv[2])
batch_size = 80 # int(sys.argv[3])
input_dim = 620 # int(sys.argv[4])

print "units=", units
print "timesteps=", timesteps
print "batch_size=", batch_size
print "input_dim=", input_dim

x = np.random.rand(timesteps, batch_size, input_dim).astype(np.float64)
wx = np.random.rand(input_dim, units).astype(np.float64) - np.random.rand(input_dim, units).astype(np.float64)
wh = np.random.rand(units, units).astype(np.float64) - np.random.rand(units, units).astype(np.float64)
b = np.zeros((batch_size, units), dtype=np.float64)
h_init = np.zeros((batch_size, units), dtype=np.float64)
o_predict = np.random.rand(timesteps, batch_size, units).astype(np.float64)

def SimpleRNN_MKL_backward():
    global x, h_init, w_x, w_h, b, units
    X = T.dtensor3('X')
    H_init = T.dmatrix('H_init')
    W_x = T.dmatrix('W_x')
    W_h = T.dmatrix('W_h')
    B = T.dmatrix('B')

    o = SimpleRNN(return_sequences=True)(X, W_x, W_h, B, H_init)
    loss = o.sum()
    gx, gwx, gwh, gb, gh_init= theano.grad(loss, [X, W_x, W_h, B, H_init])

    #f = theano.function([X, W_x, W_h, B, H_init], [o])
    f = theano.function([X, W_x, W_h, B, H_init], [loss, gx, gwx, gwh, gb, gh_init])
    # theano.printing.pydotprint(f, outfile='simple_rnn_bw.png', var_with_name_simple=True)
    return f

def SimpleRNN_theano_backward():
    X = T.dtensor3('X')
    W_x = T.dmatrix('W_x')
    W_h = T.dmatrix('W_h')
    B = T.dmatrix('B')
    Hid = T.dmatrix('hid')

    def step(x, h):
        h = T.tanh(T.dot(x, W_x) + T.dot(h, W_h) + B)
        return h

    result, updates = theano.scan(step, sequences=[X], outputs_info=Hid, name="SimpleRNN_theano")

    loss = result.sum()
    gx, gwx, gwh, gb, gh_init  = theano.grad(loss, [X, W_x, W_h, B, Hid])
    f = theano.function([X, W_x, W_h, Hid, B], [loss, gx, gwx, gwh, gb, gh_init], updates=updates)
    # theano.printing.pydotprint(f, outfile='simple_rnn_theano.png', var_with_name_simple=True)
    return f


if __name__ == '__main__':
    f_mkl = SimpleRNN_MKL_backward()
    f_theano = SimpleRNN_theano_backward()

    mkl_result = f_mkl(x, wx, wh, b, h_init)
    theano_result = f_theano(x, wx, wh, h_init, b)

    print('\n====Compare loss value===================')
    print('loss max err: %s' %(abs(mkl_result[0] - theano_result[0])))
    print('loss relative err: %s' %(abs((mkl_result[0] - theano_result[0]) / theano_result[0])))

    print('\n====Compare gradient inputs==============')
    p, q, r = np.where((mkl_result[1]-theano_result[1]) == (mkl_result[1]-theano_result[1]).max())
    print('gradient inputs max err: %s' %(abs((mkl_result[1]-theano_result[1]).max())))
    print('gradient inputs relative err: %s' %(abs((mkl_result[1]-theano_result[1]).max()/theano_result[1][p, q, r].max())))

    print('\n====Compare gradient weight for X========')
    p, q = np.where((mkl_result[2]-theano_result[2]) == (mkl_result[2]-theano_result[2]).max())
    print('gradient wegihtX max err: %s' %( abs((mkl_result[2]-theano_result[2]).max())))
    print('gradient weightX relative err: %s' %( abs((mkl_result[2]-theano_result[2]).max()/theano_result[2][p, q].max())))

    print('\n====Compare gradient weight for H========')
    p, q = np.where((mkl_result[3]-theano_result[3]) == (mkl_result[3]-theano_result[3]).max())
    print('gradient weightH max err: %s' %( abs((mkl_result[3]-theano_result[3]).max())))
    print('gradient weightH relative err: %s' %( abs((mkl_result[3]-theano_result[3]).max()/theano_result[3][p, q].max())))
    
    print('\n====Compare gradient bias================')
    p = np.where((mkl_result[4]-theano_result[4]) == (mkl_result[4]-theano_result[4]).max())
    print('gradient bias max err: %s' %( abs((mkl_result[4]-theano_result[4]).max())))
    print('gradient bias relative err: %s' %( abs((mkl_result[4]-theano_result[4]).max()/theano_result[4][p].max())))
   
    print('\n====Compare gradient hinit================')
    p = np.where((mkl_result[5]-theano_result[5]) == (mkl_result[5]-theano_result[5]).max())
    print('gradient bias max err: %s' %( abs((mkl_result[5]-theano_result[5]).max())))
    print('gradient bias relative err: %s' %( abs((mkl_result[5]-theano_result[5]).max()/theano_result[5][p].max()))) 


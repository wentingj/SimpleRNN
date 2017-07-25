import theano
from theano import tensor, gof
from theano.tensor.blas import ldflags


class SimpleRNNGrad(gof.Op):
    __props__ = ('return_sequences',)

    def __init__(self, return_sequences=False):
        self.return_sequences = return_sequences
        super(SimpleRNNGrad, self).__init__()

    def make_node(self, x, h_init, h_store, wx, wh, grads):
        x = tensor.as_tensor_variable(x)
        h_init = tensor.as_tensor_variable(h_init)
        h_store = tensor.as_tensor_variable(h_store)
        wx = tensor.as_tensor_variable(wx)
        wh = tensor.as_tensor_variable(wh)
      
        gz = tensor.as_tensor_variable(grads)

        out = [x.type(), wx.type(), wh.type(), h_init.type()]
        return gof.Apply(self, [x, h_init, h_store, wx, wh, gz], out)

    def c_headers(self):
        headers = ['<mkl.h>']
        return headers

    def c_libraries(self):
        return ldflags()

    def c_support_code_struct(self, node, name):
	if node.inputs[0].type.dtype is 'float32':
            dtype = 'float'
        elif node.inputs[0].type.dtype is 'float64':
            dtype = 'double'
        else:
            raise TypeError('Gemm: dtype %s is not supported.'
                            % (node.inputs[0].type.dtype))

        ccode = """
            %(dtype)s *h_store;
            %(dtype)s *hh_store;
	    %(dtype)s *onemhh_store;
            %(dtype)s *store;
        """ % locals()
        return ccode

    def c_init_code_struct(self, node, name, sub):
        ccode = """
	    h_store = NULL;
	    hh_store = NULL;
	    onemhh_store = NULL;
	    store = NULL;
	""" % locals()
        return ccode

    def c_cleanup_code_struct(self, node, name):
        ccode = """
	    mkl_free(h_store);
            mkl_free(hh_store);
            mkl_free(onemhh_store);
            mkl_free(store);
	"""

        return ccode

    def c_code(self, node, name, inputs, outputs, sub):
	x, h_init, h_store, wx, wh, gz = inputs
	gx, gwx, gwh, gb,  = outputs
	
	if node.inputs[0].type.dtype is 'float32':
            dtype = 's'
            d = 'float'
        elif node.inputs[0].type.dtype is 'float64':
            dtype = 'd'
            d = 'double'
        else:
            raise TypeError('Gemm: dtype %s is not supported.'
                            % (node.inputs[0].type.dtype))
        ccode = """
            int i,j,k;
	    int timesteps, input_dim, batch_size, units;
	   
            //%(d)s *dx = NULL; 
	    //npy_intp dims[3] = {0, 0, 0};
            //dims[0] = PyArray_DIMS(%(x)s)[0];
            //dims[1] = PyArray_DIMS(%(x)s)[1];
            //dims[2] = PyArray_DIMS(%(x)s)[2];

	    timesteps = PyArray_DIMS(%(x)s)[0];
	    input_dim = PyArray_DIMS(%(x)s)[2];
	    batch_size = PyArray_DIMS(%(x)s)[1];
	    units = PyArray_DIMS(%(h_init)s)[1];

	    npy_intp dims[3] = {0, 0, 0};

	    if (NULL == %(gx)s) {
                dims[0] = timesteps;
                dims[1] = batch_size;
		dims[2] = input_dim;
                %(gx)s = (PyArrayObject*) PyArray_ZEROS(3, dims, PyArray_TYPE(%(x)s), 0);
            }
	
	    if (NULL == %(gwx)s) {
                dims[0] = input_dim;
                dims[1] = units;
                %(gwx)s = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(wx)s), 0);
            }

            if (NULL == %(gwh)s) {
                dims[0] = units;
                dims[1] = units;
                %(gwh)s = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(wh)s), 0);
            }

            if (NULL == %(gb)s) {
                dims[0] = batch_size;
                dims[1] = units;
                %(gb)s = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(h_init)s), 0);
            }

            hh_store = (%(d)s*)mkl_malloc(timesteps * batch_size * units *sizeof(%(d)s), 64 );
            onemhh_store = (%(d)s*)mkl_malloc(timesteps * batch_size * units *sizeof(%(d)s), 64 );
            store = (%(d)s*)mkl_malloc(timesteps * batch_size * units * sizeof(%(d)s), 64 );
            //memset(dx, 0.0, timesteps*batch_size*input_dim*sizeof(%(d)s));

            %(d)s *tmp = (%(d)s*)mkl_malloc(batch_size * units * sizeof(%(d)s), 64 );
            %(d)s *tmp2 = (%(d)s*)mkl_malloc(batch_size * input_dim * sizeof(%(d)s), 64 );
            memset(tmp2, 0.0, batch_size * input_dim * sizeof(%(d)s));
            
	   
	    //backward
            for(i=0;i<timesteps;i++){
                v%(dtype)sMul( batch_size*units, (%(d)s*) PyArray_DATA(%(h_store)s)+i*batch_size * units, (%(d)s*) PyArray_DATA(%(h_store)s)+i*batch_size * units, hh_store+i*batch_size * units);
            }
            for(i=0;i<timesteps*batch_size*units;i++){
                onemhh_store[i] = 1-hh_store[i];
                //store[i] = -1*log_hreal_store[i]*onemhh_store[i];
                store[i] = ((%(d)s*) PyArray_DATA(%(gz)s))[i]*onemhh_store[i];
            }

            for(i=timesteps-1; i>=0; i--){
                for(j=timesteps-1; j>i; j--){
                    for(int q=0;q<batch_size*units;q++){
                        tmp[q] = store[j*batch_size*units+q];
                    }
                    cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, units, units, 1.0,tmp, units, (%(d)s*) PyArray_DATA(%(wh)s), units, 0.0, store + j*batch_size*units, units);
                    v%(dtype)sMul(batch_size*units, store + j*batch_size*units, onemhh_store + i*batch_size*units, store + j*batch_size*units);
                }
		
                for(k=timesteps-1; k>=i;k--){
                    cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, input_dim, units, 1.0,  store + k*batch_size*units, units, (%(d)s*) PyArray_DATA(%(wx)s), units, 1.0, tmp2, input_dim);
                    cblas_%(dtype)sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, input_dim, units, batch_size, 1.0, (%(d)s*) PyArray_DATA(%(x)s) + i*batch_size*input_dim, input_dim, store + k*batch_size*units, units, 1.0, (%(d)s*) PyArray_DATA(%(gwx)s), units);
                    //m, n, k, alpha, h_tm1, m, W_zh, n, beta, z_t, n);
            	    if(i==0)
            	        cblas_%(dtype)sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, units, units, batch_size,1.0, (%(d)s*) PyArray_DATA(%(h_init)s), units, store + k*batch_size*units, units, 1.0, (%(d)s*) PyArray_DATA(%(gwh)s), units);
            	    else
            	        cblas_%(dtype)sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, units, units, batch_size,1.0, (%(d)s*) PyArray_DATA(%(h_store)s) + (i-1)*batch_size * units, units, store + k*batch_size*units, units, 1.0, (%(d)s*) PyArray_DATA(%(gwh)s), units);
		    v%(dtype)sAdd(batch_size * units, store + k*batch_size*units, (%(d)s*) PyArray_DATA(%(gb)s), (%(d)s*) PyArray_DATA(%(gb)s));
		}
                v%(dtype)sAdd(batch_size*input_dim, (%(d)s*) PyArray_DATA(%(gx)s)+i*batch_size*input_dim, tmp2, (%(d)s*) PyArray_DATA(%(gx)s)+i*batch_size*input_dim);
                memset(tmp2, 0.0, batch_size * input_dim * sizeof(%(d)s));
            }

            mkl_free(tmp);
            mkl_free(tmp2);

	""" % locals()
        return ccode

    def c_code_cache_version(self):
        return (1, 0, 0)

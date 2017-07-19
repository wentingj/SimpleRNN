import theano
from theano import tensor, gof
from theano.tensor.blas import ldflags


class SimpleRNN_bw(gof.Op):
    __props__ = ()

    def __init__(self):
	super(SimpleRNN_bw, self).__init__()

    def make_node(self, x, wx, wh, b, h_init, h_real):
        x = tensor.as_tensor_variable(x)
        wx = tensor.as_tensor_variable(wx)
        wh = tensor.as_tensor_variable(wh)
        b = tensor.as_tensor_variable(b)
        h_init = tensor.as_tensor_variable(h_init)
        h_real = tensor.as_tensor_variable(h_real)
        out = [h_real.type()]
        return gof.Apply(self, [x, wx, wh, b, h_init, h_real], out)

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
            float *h_store;
            float *hh_store;
	    float *onemhh_store;
	    float *log_hreal_store;
            float *store;
        """ % locals()
        return ccode

    def c_init_code_struct(self, node, name, sub):
        ccode = """
	    h_store = NULL;
	    hh_store = NULL;
	    onemhh_store = NULL;
	    log_hreal_store = NULL;
	    store = NULL;
	""" % locals()
        return ccode

    def c_cleanup_code_struct(self, node, name):
        ccode = """
	    mkl_free(h_store);
            mkl_free(hh_store);
            mkl_free(onemhh_store);
            mkl_free(log_hreal_store);
            mkl_free(store);
	"""

        return ccode

    def c_code(self, node, name, inputs, outputs, sub):
	x, wx, wh, b, h_init, h_real = inputs
	o, = outputs
	
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
	   
            %(d)s *dx = NULL; 
	    npy_intp dims[3] = {0, 0, 0};
            dims[0] = PyArray_DIMS(%(x)s)[0];
            dims[1] = PyArray_DIMS(%(x)s)[1];
            dims[2] = PyArray_DIMS(%(x)s)[2];

            if (! %(o)s) {
                %(o)s = (PyArrayObject*) PyArray_ZEROS(3, dims, PyArray_TYPE(%(x)s), 0);
            }

            dx = (%(d)s*)PyArray_DATA(%(o)s);

	    timesteps = PyArray_DIMS(%(x)s)[0];
	    input_dim = PyArray_DIMS(%(x)s)[2];
	    batch_size = PyArray_DIMS(%(x)s)[1];
	    units = PyArray_DIMS(%(h_init)s)[1];

            h_store = (float*)mkl_malloc(timesteps * batch_size * units * sizeof(float), 64 );
            hh_store = (float*)mkl_malloc(timesteps * batch_size * units *sizeof(float), 64 );
            onemhh_store = (float*)mkl_malloc(timesteps * batch_size * units *sizeof(float), 64 );
            log_hreal_store = (float*)mkl_malloc(timesteps * batch_size * units* sizeof(float), 64 );
            store = (float*)mkl_malloc(timesteps * batch_size * units * sizeof(float), 64 );
            memset(dx, 0.0, timesteps*batch_size*input_dim*sizeof( float ));

            //store forward h in h_store
            float *tmp = (float*)mkl_malloc(batch_size * units * sizeof(float), 64 );
            float *tmp2 = (float*)mkl_malloc(batch_size * input_dim * sizeof(float), 64 );
            memset(tmp2, 0.0, batch_size * input_dim * sizeof(float));
            for(i=0; i<timesteps; i++){
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, units, input_dim, 1.0, (%(d)s*) PyArray_DATA(%(x)s) + i * batch_size * input_dim, input_dim, (%(d)s*) PyArray_DATA(%(wx)s),  units, 0.0, tmp, units);
                if(i == 0)
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, units, units, 1.0, (%(d)s*) PyArray_DATA(%(h_init)s), units, (%(d)s*) PyArray_DATA(%(wh)s), units, 1.0, tmp, units);
                else
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, units, units, 1.0, h_store+(i-1)*batch_size * units, units, (%(d)s*) PyArray_DATA(%(wh)s), units, 1.0, tmp, units);
                vsAdd(batch_size*units, tmp, (%(d)s*) PyArray_DATA(%(b)s), tmp);
                vsTanh(batch_size*units, tmp, h_store+i*batch_size * units);
            }
            //backward
            for(i=0;i<timesteps;i++){
                vsLn( batch_size*units, (%(d)s*) PyArray_DATA(%(h_real)s)+i*batch_size * units, log_hreal_store+i*batch_size * units );
                vsMul( batch_size*units, h_store+i*batch_size * units, h_store+i*batch_size * units, hh_store+i*batch_size * units);
            }
            for(i=0;i<timesteps*batch_size*units;i++){
                onemhh_store[i] = 1-hh_store[i];
                store[i] = -1*log_hreal_store[i]*onemhh_store[i];
            }

            for(i=timesteps-1; i>=0; i--){
                for(j=timesteps-1; j>i; j--){
                    for(int q=0;q<batch_size*units;q++){
                        tmp[q] = store[j*batch_size*units+q];
                    }
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, units, units, 1.0,tmp, units, (%(d)s*) PyArray_DATA(%(wh)s), units, 0.0, store + j*batch_size*units, units);
                    vsMul(batch_size*units, store + j*batch_size*units, onemhh_store + i*batch_size*units, store + j*batch_size*units);
                }
		
                for(k=timesteps-1; k>=i;k--){
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, input_dim, units, 1.0,  store + k*batch_size*units, units, (%(d)s*) PyArray_DATA(%(wx)s), units, 1.0, tmp2, input_dim);
                }
                vsAdd(batch_size*input_dim, dx+i*batch_size*input_dim, tmp2, dx+i*batch_size*input_dim);
                memset(tmp2, 0.0, batch_size * input_dim * sizeof(float));
            }

            mkl_free(tmp);
            mkl_free(tmp2);

	""" % locals()
        return ccode

    def c_code_cache_version(self):
        return (1, 0, 0)

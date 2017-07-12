import theano
from theano import tensor, gof
from theano.tensor.blas import ldflags


class SimpleRNN(gof.Op):
    __props__ = ()

    def __init__(self):
	super(SimpleRNN, self).__init__()

    def make_node(self, x, h_init, w_x, w_h, b):
	x = tensor.as_tensor_variable(x)
        h_init = tensor.as_tensor_variable(h_init)
        w_x = tensor.as_tensor_variable(w_x)
        w_h = tensor.as_tensor_variable(w_h)
        b = tensor.as_tensor_variable(b)
        out = [h_init.type()]
        return gof.Apply(self, [x, h_init, w_x, w_h, b], out)

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
	    %(dtype)s** A;
            %(dtype)s** B;
            %(dtype)s** C;

            MKL_INT    m_g[1];
            MKL_INT    k_g[1];
            MKL_INT    n_g[1];
            MKL_INT    lda_g[1];
            MKL_INT    ldb_g[1];
            MKL_INT    ldc_g[1];

            CBLAS_TRANSPOSE    transA_g[1];
            CBLAS_TRANSPOSE    transB_g[1];

            %(dtype)s  alpha_g[1];
            %(dtype)s  beta_g[1];
            MKL_INT    size_per_grp[1];
        """ % locals()
        return ccode

    def c_init_code_struct(self, node, name, sub):
        ccode = """
            A = NULL;
            B = NULL;
            C = NULL;

            m_g[0] = 0;
            k_g[0] = 0;
            n_g[0] = 0;

            lda_g[0] = 0;
            ldb_g[0] = 0;
            ldc_g[0] = 0;

            transA_g[0] = CblasNoTrans;
            transB_g[0] = CblasNoTrans;

            alpha_g[0] = 1.0;
            beta_g[0] = 1.0;
            size_per_grp[0] = 1;
	""" % locals()
        return ccode

    def c_cleanup_code_struct(self, node, name):
        ccode = """
            if (A) {
                free (A);
                A = NULL;
            }

            if (B) {
                free (B);
                B = NULL;
            }

            if (C) {
                free (C);
                C =NULL;
            }
	"""

        return ccode

    def c_code(self, node, name, inputs, outputs, sub):
	x, h_init, w_x, w_h, b = inputs
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
	print locals()
        ccode = """
	    int i,j;
	    int timesteps, input_dim, batch_size, units;
	    timesteps = PyArray_DIMS(%(x)s)[0];
	    input_dim = PyArray_DIMS(%(x)s)[1];
	    batch_size = PyArray_DIMS(%(x)s)[2];
	    units = PyArray_DIMS(%(h_init)s)[0];

            m_g[0] = units;
            k_g[0] = input_dim;
            n_g[0] = batch_size;
            lda_g[0] = k_g[0];
            ldb_g[0] = n_g[0];
            ldc_g[0] = n_g[0];
            
            size_per_grp[0] = timesteps;

            if (A == NULL)
                A = (%(d)s**)malloc(timesteps * sizeof (%(d)s*));

            if (B == NULL)
                B = (%(d)s**)malloc(timesteps * sizeof (%(d)s*));

            if (C == NULL)
                C = (%(d)s**)malloc(timesteps * sizeof (%(d)s*));

            for (i = 0 ; i < timesteps; i ++) {
                A[i] = (%(d)s*) PyArray_DATA(%(w_x)s);
                B[i] = (%(d)s*) PyArray_DATA(%(x)s) + i * batch_size * input_dim;
                C[i] = (%(d)s*) PyArray_DATA(%(b)s) + i * batch_size * units;
            }
            //xW+b
            cblas_%(dtype)sgemm_batch (
                        CblasRowMajor,
                        transA_g,
                        transB_g,
                        m_g,
                        n_g,
                        k_g,
                        alpha_g,
                        A,
                        lda_g,
                        B,
                        ldb_g,
                        beta_g,
                        C,
                        ldc_g,
                        1,
                        size_per_grp);

	    int sz = units * batch_size;
            %(d)s *h_tm1 = NULL;

            npy_intp dims[2] = {0, 0}; 
	    dims[0] = PyArray_DIMS(%(h_init)s)[0];
            dims[1] = PyArray_DIMS(%(h_init)s)[1];
 
	    if (! %(o)s) {
                %(o)s = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(h_init)s), 0);
            }

	    h_tm1 = (%(d)s*)PyArray_DATA(%(o)s);
            for(i = 0; i < sz; i++){
                h_tm1[i] = ((%(d)s*) PyArray_DATA(%(h_init)s))[i];
            }
            for (i = 0; i < timesteps; i ++) {
		cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        units, batch_size, units, 1.0, (%(d)s*) PyArray_DATA(%(w_h)s), units, h_tm1, batch_size, 1.0, C[i], batch_size);
	        v%(dtype)sTanh(sz, C[i], h_tm1);
            }
	
	""" % locals()
        return ccode

    def c_code_cache_version(self):
        return (1, 0, 0)

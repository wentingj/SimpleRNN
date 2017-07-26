import theano
from theano import tensor, gof
from theano.tensor.blas import ldflags
from mkl_simplernn_backward import SimpleRNNGrad
from theano.gradient import DisconnectedType

class SimpleRNN(gof.Op):
    __props__ = ('return_sequences',)

    def __init__(self, return_sequences=False):
	self.return_sequences = return_sequences
	super(SimpleRNN, self).__init__()

    def make_node(self, *inputs):
	inp = list(map(tensor.as_tensor_variable, inputs))
	#x = tensor.as_tensor_variable(x)
        #h_init = tensor.as_tensor_variable(h_init)
        #w_x = tensor.as_tensor_variable(w_x)
        #w_h = tensor.as_tensor_variable(w_h)
        #b = tensor.as_tensor_variable(b)

        assert inp[0].ndim is 3 #x
        assert inp[1].ndim is 2 #w_x
        assert inp[2].ndim is 2 #w_h
        assert inp[3].ndim is 2 #b
        assert inp[4].ndim is 2 #h_init

	if self.return_sequences:
            out = [inp[0].type()]
        else:
            out = [inp[1].type()]
        return gof.Apply(self, inp, out)

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
	    size_t timesteps;
	    size_t input_dim;
	    size_t batch_size;
	    size_t units;

	    %(dtype)s** A;
            %(dtype)s** B;
            %(dtype)s** C;

	    %(dtype)s* h_tm1;
	    %(dtype)s* b_3d;

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
	    timesteps = 0;
	    input_dim = 0;
	    batch_size = 0;
	    units = 0;            

	    A = NULL;
            B = NULL;
            C = NULL;
            h_tm1 = NULL;
	    b_3d = NULL;

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
                mkl_free (A);
                A = NULL;
            }

            if (B) {
                mkl_free (B);
                B = NULL;
            }

            if (C) {
                mkl_free (C);
                C = NULL;
            }

	    if (h_tm1) {
                mkl_free (h_tm1);
                h_tm1 = NULL;
            }

	    if (b_3d) {
		mkl_free (b_3d);
		b_3d = NULL;
	    }
	"""

        return ccode

    def c_code(self, node, name, inputs, outputs, sub):
	x, w_x, w_h, b, h_init = inputs
	if self.return_sequences:
            return_sequences = 1
        else:
            return_sequences = 0
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
	#print locals()
        ccode = """
	    int i = 0;
	    timesteps = PyArray_DIMS(%(x)s)[0];
	    input_dim = PyArray_DIMS(%(x)s)[2];
	    batch_size = PyArray_DIMS(%(x)s)[1];
	    units = PyArray_DIMS(%(h_init)s)[1];

	    %(d)s* b_ptr = NULL;

	    PyArrayObject* b_src = NULL;
	    if (!PyArray_IS_C_CONTIGUOUS(%(b)s)) {
                    printf(\"Warning: Need convert bias to C-Contiguous\\n\");
                    b_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(b)s,
                                                PyArray_TYPE(%(b)s),
                                                PyArray_NDIM(%(b)s),
                                                PyArray_NDIM(%(b)s));
                    if (!b_src) {
                        PyErr_SetString(PyExc_RuntimeError, \"GRU: fail to case bias to contiguous array\");
                        goto simplernn_fail;
                    }
                    b_ptr = (%(d)s*) PyArray_DATA(b_src);
                } else {
                    b_ptr = (%(d)s*) PyArray_DATA(%(b)s);
            }            
	
	    // check Dot(x, w_x)
            if (PyArray_DIMS(%(w_x)s)[0] != PyArray_DIMS(%(x)s)[2]) {
                PyErr_SetString(PyExc_RuntimeError, \"GRU: dot x,w_x size error\");
                goto simplernn_fail;
            }
	    // check Dot(h_init, w_h)
	    if (PyArray_DIMS(%(w_h)s)[0] != PyArray_DIMS(%(h_init)s)[1]) {
                PyErr_SetString(PyExc_RuntimeError, \"GRU: dot h_init,w_h size error\");
                goto simplernn_fail;
            }
	    // check b
            if (PyArray_DIMS(%(b)s)[1] != PyArray_DIMS(%(w_h)s)[1]) {
                PyErr_SetString(PyExc_RuntimeError, \"GRU: b size error\");
                goto simplernn_fail;
            }

	    if (NULL == b_3d) {
                b_3d = (%(d)s*) mkl_malloc(timesteps * batch_size * units * sizeof (%(d)s), 64);
            }

	    for (i = 0; i < timesteps; i++){
	    	memcpy((void*)(b_3d) + i*batch_size*units* sizeof (%(d)s), (void*)b_ptr,  batch_size*units* sizeof (%(d)s));
	    }

	    m_g[0] = batch_size;
            k_g[0] = input_dim;
            n_g[0] = units;
            lda_g[0] = k_g[0];
            ldb_g[0] = n_g[0];
            ldc_g[0] = n_g[0];
            
            size_per_grp[0] = timesteps;

            if (A == NULL)
                A = (%(d)s**)mkl_malloc(timesteps * sizeof (%(d)s*), 64);

            if (B == NULL)
                B = (%(d)s**)mkl_malloc(timesteps * sizeof (%(d)s*), 64);

            if (C == NULL)
                C = (%(d)s**)mkl_malloc(timesteps * sizeof (%(d)s*), 64);

	    if (A == NULL || B == NULL || C == NULL) {
	      	printf( "\\n ERROR: Can't allocate memory for matrices. Aborting... \\n");
      		mkl_free(A);
      		mkl_free(B);
      		mkl_free(C);
      		return 1;
    	    }

            for (i = 0 ; i < timesteps; i ++) {
                A[i] = (%(d)s*) PyArray_DATA(%(x)s) + i * batch_size * input_dim;
                B[i] = (%(d)s*) PyArray_DATA(%(w_x)s);
                C[i] = (%(d)s*) b_3d + i * batch_size * units;
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
            //%(d)s *h_tm1 = NULL;

            npy_intp dims[3] = {0, 0, 0};
            if (NULL == %(o)s) {
                if (%(return_sequences)s) {
                    dims[0] = timesteps;
                    dims[1] = batch_size;
                    dims[2] = units;
                    %(o)s = (PyArrayObject*) PyArray_ZEROS(3, dims, PyArray_TYPE(%(x)s), 0);
                } else {
                    dims[0] = batch_size;
                    dims[1] = units;
                    %(o)s = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(x)s), 0);
                }
            }
 
	    //h_tm1 = (%(d)s*)PyArray_DATA(%(o)s);
            if (NULL == h_tm1) {
                h_tm1 = (%(d)s*) mkl_malloc(batch_size * units * sizeof (%(d)s), 64);
            }

            if (NULL == h_tm1) {
                PyErr_SetString(PyExc_MemoryError, \"SimpleRNN: create buffer for hidden state failed\");
                goto simplernn_fail;
            }
            for(i = 0; i < sz; i++){
                h_tm1[i] = ((%(d)s*) PyArray_DATA(%(h_init)s))[i];
            }
            for (i = 0; i < timesteps; i ++) {
		cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        batch_size, units, units, 1.0, h_tm1, units, (%(d)s*) PyArray_DATA(%(w_h)s), units, 1.0, C[i], units);
	        v%(dtype)sTanh(sz, C[i], h_tm1);
                
                if (%(return_sequences)s) {
                    memcpy(((char*)PyArray_DATA(%(o)s)) + i * sz * sizeof (%(d)s),
                            (char*)h_tm1,
                            sz * sizeof (%(d)s));
                } else {
                    if (i == timesteps - 1) {
                        memcpy(((char*)PyArray_DATA(%(o)s)),
                                (char*)h_tm1,
                                sz * sizeof (%(d)s));
                    }
                }     
	    }

            simplernn_fail:
            Py_XDECREF(b_src);	
	""" % locals()
        return ccode
    
    def grad(self, inp, grads):
	x, wx, wh, b, h_init, = inp[0:5]
	gz, = grads
        #disc = [DisconnectedType()() for i in inp[4:]]
        
	h = SimpleRNN(self.return_sequences)(x, wx, wh, b, h_init)
        gradX, gradW, gradU, gradB, gradHinit = SimpleRNNGrad(self.return_sequences)(x, h_init, h, wx, wh, gz)
	return [gradX, gradW, gradU, gradB, gradHinit]
	#return [gradX, gradW, gradU, gradB] + disc

    def c_code_cache_version(self):
        return (1, 0, 0)

    #def connect_pattern(self, node):
    #    return [[1], [1], [1], [1], [0]]

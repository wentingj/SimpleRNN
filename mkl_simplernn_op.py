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

        assert x.ndim is 3
        assert h_init.ndim is 2
        assert w_x.ndim is 2
        assert w_h.ndim is 2
        assert b.ndim is 3

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
	    size_t timesteps;
	    size_t input_dim;
	    size_t batch_size;
	    size_t units;

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
	    timesteps = 0;
	    input_dim = 0;
	    batch_size = 0;
	    units = 0;            

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
                mkl_free (A);
                A = NULL;
            }

            if (B) {
                mkl_free (B);
                B = NULL;
            }

            if (C) {
                mkl_free (C);
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
	    int i = 0;
	    timesteps = PyArray_DIMS(%(x)s)[0];
	    input_dim = PyArray_DIMS(%(x)s)[1];
	    batch_size = PyArray_DIMS(%(x)s)[2];
	    units = PyArray_DIMS(%(h_init)s)[0];

	    %(d)s* x_ptr = NULL;
            %(d)s* w_x_ptr = NULL;
            %(d)s* w_h_ptr = NULL;
	    %(d)s* b_ptr = NULL;

	    PyArrayObject* x_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(x)s)) {
                printf(\"Warning: Need convert x to C-Contiguous\\n\");
                x_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(x)s,
                                            PyArray_TYPE(%(x)s),
                                            PyArray_NDIM(%(x)s),
                                            PyArray_NDIM(%(x)s));
                if (!x_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRU: fail to cast x to contiguous array\");
                    goto gru_fail;
                }
                x_ptr = (%(d)s*) PyArray_DATA(x_src);
            } else {
                x_ptr = (%(d)s*) PyArray_DATA(%(x)s);
            }

	    PyArrayObject* w_x_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(w_x)s)) {
                printf(\"Warning: Need convert w_x to C-Contiguous\\n\");
                w_x_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(w_x)s,
                                            PyArray_TYPE(%(w_x)s),
                                            PyArray_NDIM(%(w_x)s),
                                            PyArray_NDIM(%(w_x)s));
                if (!w_x_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRU: fail to cast w_x to contiguous array\");
                    goto gru_fail;
                }
                w_x_ptr = (%(d)s*) PyArray_DATA(w_x_src);
            } else {
                w_x_ptr = (%(d)s*) PyArray_DATA(%(w_x)s);
            }

	    PyArrayObject* w_h_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(w_h)s)) {
                printf(\"Warning: Need convert w_h to C-Contiguous\\n\");
                w_h_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(w_h)s,
                                            PyArray_TYPE(%(w_h)s),
                                            PyArray_NDIM(%(w_h)s),
                                            PyArray_NDIM(%(w_h)s));
                if (!w_h_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRU: fail to cast w_h to contiguous array\");
                    goto gru_fail;
                }
                w_h_ptr = (%(d)s*) PyArray_DATA(w_h_src);
            } else {
                w_h_ptr = (%(d)s*) PyArray_DATA(%(w_h)s);
            }

	    PyArrayObject* b_src = NULL;
	    if (!PyArray_IS_C_CONTIGUOUS(%(b)s)) {
                    printf(\"Warning: Need convert bias to C-Contiguous\\n\");
                    b_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(b)s,
                                                PyArray_TYPE(%(b)s),
                                                PyArray_NDIM(%(b)s),
                                                PyArray_NDIM(%(b)s));
                    if (!b_src) {
                        PyErr_SetString(PyExc_RuntimeError, \"GRU: fail to case bias to contiguous array\");
                        goto gru_fail;
                    }
                    b_ptr = (%(d)s*) PyArray_DATA(b_src);
                } else {
                    b_ptr = (%(d)s*) PyArray_DATA(%(b)s);
            }            
	
	    // check Dot(w_x, x)
            if (PyArray_DIMS(%(w_x)s)[1] != PyArray_DIMS(%(x)s)[1]) {
                PyErr_SetString(PyExc_RuntimeError, \"GRU: w_x * x size error\");
                goto gru_fail;
            }
	    // check Dot(w_h, h_init)
	    if (PyArray_DIMS(%(w_h)s)[1] != PyArray_DIMS(%(h_init)s)[0]) {
                PyErr_SetString(PyExc_RuntimeError, \"GRU: w_h * h_init size error\");
                goto gru_fail;
            }
	    // check b
            if (PyArray_DIMS(%(b)s)[0] != PyArray_DIMS(%(x)s)[0] || PyArray_DIMS(%(b)s)[1] != PyArray_DIMS(%(w_x)s)[0] || PyArray_DIMS(%(b)s)[2] != PyArray_DIMS(%(x)s)[2]) {
                PyErr_SetString(PyExc_RuntimeError, \"GRU: b size error\");
                goto gru_fail;
            }

	    m_g[0] = units;
            k_g[0] = input_dim;
            n_g[0] = batch_size;
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

            gru_fail:
            Py_XDECREF(x_src);
            Py_XDECREF(w_x_src);
            Py_XDECREF(w_h_src);
            Py_XDECREF(b_src);	
	""" % locals()
        return ccode

    def c_code_cache_version(self):
        return (1, 0, 0)

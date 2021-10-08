import numpy as np
import tvm
from numpy_dlpack import np_to_nd, nd_to_np


def test_np_to_nd():
    # Test converting numpy to tvm ndarray
    print("### Testing np_to_nd")
    array = np.random.normal(size=[10, 10])
    array_ref = array.copy()
    nd_array = np_to_nd(array)
    del array
    array_back = nd_array.numpy()
    np.testing.assert_equal(actual=array_ref, desired=array_back)
    del nd_array


def test_nd_to_np():
    # Test converting tvm ndarray to numpy
    print("### Testing nd_to_np")
    array = tvm.nd.array(np.random.normal(size=[10, 10]))
    array_ref = array.numpy()
    np_array = nd_to_np(array)
    del array
    np.testing.assert_equal(actual=array_ref, desired=np_array)
    del np_array

import time
for i in range(10000):
    test_np_to_nd()
    #test_nd_to_np()

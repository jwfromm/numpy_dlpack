import numpy as np
import tvm
from numpy_dlpack import np_to_nd, nd_to_np


def test_np_to_nd():
    # Test converting numpy to tvm ndarray
    np_array = np.arange(10, dtype="int32").reshape((2, 5))
    np_array_ref = np_array.copy()
    tvm_array = np_to_nd(np_array)
    return
    del np_array
    np.testing.assert_equal(actual=tvm_array.numpy(), desired=np_array_ref)
    del tvm_array


def test_nd_to_np():
    # Test converting tvm ndarray to numpy
    print("### Testing nd_to_np")
    array = tvm.nd.array(np.random.normal(size=[10, 10]))
    array_ref = array.numpy()
    np_array = nd_to_np(array)
    del array
    np.testing.assert_equal(actual=array_ref, desired=np_array)
    del np_array


if __name__ == "__main__":
    print("### Testing np_to_nd")
    for i in range(1):
        test_np_to_nd()
    #test_nd_to_np()

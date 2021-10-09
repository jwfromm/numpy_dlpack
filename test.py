import ctypes
import numpy as np
import tvm
from dlpack import from_numpy, to_numpy


def test_from_numpy():
    np_array = np.random.normal(size=[10, 10])
    np_array_ref = np_array.copy()
    tvm_array = from_numpy(np_array, tvm.nd.from_dlpack)
    del np_array
    np.testing.assert_equal(actual=tvm_array.numpy(), desired=np_array_ref)
    del tvm_array


def test_to_numpy():
    tvm_array = tvm.nd.array(np.random.normal(size=[10, 10]))
    np_array_ref = tvm_array.numpy()
    np_array = to_numpy(tvm_array)
    del tvm_array
    np.testing.assert_equal(actual=np_array, desired=np_array_ref)
    del np_array


if __name__ == "__main__":
    print("### Testing from_numpy")
    for i in range(10000):
        test_from_numpy()
    print("### Testing to_numpy")
    for i in range(10000):
        test_to_numpy()

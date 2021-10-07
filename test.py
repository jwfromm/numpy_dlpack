import numpy as np
import tvm
import tvm.testing
from numpy_dlpack import np_to_nd, nd_to_np

# test converting numpy to tvm ndarray
array = np.random.normal(size=[10, 10])
nd_array = np_to_nd(array)
print(nd_array.numpy())
print(array.shape)
exit()

# test converting ndarray to numpy
array = tvm.nd.array(np.random.normal(size=[10, 10]))
np_array = nd_to_np(array)
print(np_array)
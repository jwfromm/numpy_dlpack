import ctypes
import dlpack
import tvm
import numpy as np

DLManagedTensorPointer = ctypes.POINTER(dlpack.DLManagedTensor)
PyObjPtr = ctypes.POINTER(ctypes.py_object)

_libmain = ctypes.PyDLL("./libmain.so")
libmain_Debug = _libmain.Debug
libmain_Debug.restype = ctypes.c_int
libmain_Debug.argtypes = [ctypes.py_object, ctypes.py_object]
libmain_DebugDLManagedTensor = _libmain.DebugDLManagedTensor
libmain_DebugDLManagedTensor.restype = ctypes.c_int
libmain_DebugDLManagedTensor.argtypes = [DLManagedTensorPointer]


ctypes.pythonapi.PyMem_RawMalloc.restype = ctypes.c_void_p
ctypes.pythonapi.PyMem_RawFree.argtypes = [ctypes.c_void_p]







class NpHolder:

    def __init__(self, np_array: np.ndarray) -> None:
        self.np_array = np_array
        self.data = np_array.ctypes.data_as(ctypes.c_void_p)
        self.shape = np_array.ctypes.shape_as(ctypes.c_int64)
        self.strides = np_array.ctypes.strides_as(ctypes.c_int64)
        for i in range(np_array.ndim):
            self.strides[i] //= np_array.itemsize




def np_to_nd(array: np.ndarray) -> tvm.nd.NDArray:
    holder = NpHolder(array)
    dl_managed_tensor: dlpack.DLManagedTensor = _alloc_dl_managed_tensor()
    dl_managed_tensor.dl_tensor.data = holder.data
    dl_managed_tensor.dl_tensor.device = dlpack.DLDevice(1, 0)
    dl_managed_tensor.dl_tensor.ndim = array.ndim
    dl_managed_tensor.dl_tensor.dtype = dlpack.DLDataType.TYPE_MAP[
        str(array.dtype)
    ]
    dl_managed_tensor.dl_tensor.shape = holder.shape
    dl_managed_tensor.dl_tensor.strides = holder.strides
    dl_managed_tensor.dl_tensor.byte_offset = 0
    make_manager_ctx(holder, dl_managed_tensor)
    dl_managed_tensor.deleter = _numpy_array_deleter
    libmain_DebugDLManagedTensor(ctypes.byref(dl_managed_tensor))
    pycapsule = ctypes.pythonapi.PyCapsule_New(
        ctypes.byref(dl_managed_tensor),
        _c_str_dltensor,
        _numpy_pycapsule_deleter,
    )
    result = tvm.nd.from_dlpack(pycapsule)
    return result



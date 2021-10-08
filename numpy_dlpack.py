import ctypes
import dlpack
import tvm
import numpy as np

DLManagedTensorPointer = ctypes.POINTER(dlpack.DLManagedTensor)
_c_str_dltensor = b"dltensor"

def _alloc_dl_managed_tensor():
    array = ctypes.create_string_buffer(ctypes.sizeof(dlpack.DLManagedTensor))
    return ctypes.cast(array, DLManagedTensorPointer)

@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _numpy_array_deleter(handle: ctypes.c_void_p) -> None:
    dl_managed_tensor: DLManagedTensorPointer = ctypes.cast(
        handle, DLManagedTensorPointer
    )
    void_p = dl_managed_tensor.contents.manager_ctx
    py_obj = ctypes.cast(void_p, ctypes.py_object)
    ctypes.pythonapi.Py_DecRef(py_obj)
    del dl_managed_tensor


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _numpy_pycapsule_deleter(handle: ctypes.c_void_p) -> None:
    pycapsule: ctypes.py_object = ctypes.cast(handle, ctypes.py_object)
    if ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _c_str_dltensor):
        ptr = ctypes.pythonapi.PyCapsule_GetPointer(pycapsule, _c_str_dltensor)
        ptr = ctypes.cast(ptr, ctypes.c_void_p)
        _numpy_array_deleter(ptr)
        ctypes.pythonapi.PyCapsule_SetDestructor(pycapsule, None)


def np_to_nd(array: np.ndarray) -> tvm.nd.NDArray:
    def make_manager_ctx(dl_managed_tensor: DLManagedTensorPointer):
        py_obj = ctypes.py_object(array)
        ctypes.pythonapi.Py_IncRef(py_obj)
        dl_managed_tensor.contents.manager_ctx = ctypes.c_void_p.from_buffer(py_obj)

    dl_managed_tensor: DLManagedTensorPointer = _alloc_dl_managed_tensor()#libmain.AllocDLManagedTensor()
    dl_managed_tensor.contents.dl_tensor.data = array.ctypes.data_as(ctypes.c_void_p)
    dl_managed_tensor.contents.dl_tensor.device = dlpack.DLDevice(1, 0)
    dl_managed_tensor.contents.dl_tensor.ndim = array.ndim
    dl_managed_tensor.contents.dl_tensor.dtype = dlpack.DLDataType.TYPE_MAP[
        str(array.dtype)
    ]
    dl_managed_tensor.contents.dl_tensor.shape = array.ctypes.shape_as(ctypes.c_int64)
    dl_managed_tensor.contents.dl_tensor.strides = array.ctypes.strides_as(
        ctypes.c_int64
    )
    for i in range(array.ndim):
        dl_managed_tensor.contents.dl_tensor.strides[i] //= array.itemsize
    dl_managed_tensor.contents.dl_tensor.byte_offset = 0
    make_manager_ctx(dl_managed_tensor)
    dl_managed_tensor.contents.deleter = _numpy_array_deleter
    pycapsule = ctypes.pythonapi.PyCapsule_New(
        dl_managed_tensor,
        _c_str_dltensor,
        _numpy_pycapsule_deleter,
    )
    return tvm.nd.from_dlpack(pycapsule)


class Holder:
    def __init__(self, array_interface, pycapsule) -> None:
        self.__array_interface__ = array_interface
        self._pycapsule = pycapsule


def nd_to_np(array: tvm.nd.NDArray) -> np.ndarray:
    pycapsule = array.to_dlpack()
    pycapsule = ctypes.py_object(pycapsule)
    assert ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _c_str_dltensor)
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(pycapsule, _c_str_dltensor)
    ptr = ctypes.cast(ptr, DLManagedTensorPointer)
    holder = Holder(ptr.contents.__array_interface__, pycapsule)
    return np.ctypeslib.as_array(holder)

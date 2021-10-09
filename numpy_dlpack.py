import ctypes
import dlpack
import tvm
import numpy as np

_c_str_dltensor = b"dltensor"
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


def _alloc_dl_managed_tensor():
    size = ctypes.sizeof(dlpack.DLManagedTensor)
    size = ctypes.c_size_t(size)
    address = ctypes.pythonapi.PyMem_RawMalloc(size)
    print("address =", hex(address))
    result = dlpack.DLManagedTensor.from_address(address)
    return result


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _numpy_array_deleter(address: ctypes.c_void_p) -> None:
    print("_numpy_array_deleter:", hex(address))
    dl_managed_tensor = dlpack.DLManagedTensor.from_address(address)
    libmain_DebugDLManagedTensor(ctypes.byref(dl_managed_tensor))
    py_obj_ptr = ctypes.cast(dl_managed_tensor.manager_ctx, PyObjPtr)
    py_obj = py_obj_ptr.contents
    libmain_Debug(py_obj, ctypes.py_object(NpHolder))
    ctypes.pythonapi.Py_DecRef(py_obj)
    ctypes.pythonapi.Py_DecRef(ctypes.py_object(py_obj_ptr))
    ctypes.pythonapi.PyMem_RawFree(address)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _numpy_pycapsule_deleter(handle: ctypes.c_void_p) -> None:
    print("_numpy_pycapsule_deleter:")
    pycapsule: ctypes.py_object = ctypes.cast(handle, ctypes.py_object)
    if ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _c_str_dltensor):
        dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(pycapsule, _c_str_dltensor)
        dl_managed_tensor = ctypes.cast(dl_managed_tensor, DLManagedTensorPointer)
        _numpy_array_deleter(dl_managed_tensor)
        ctypes.pythonapi.PyCapsule_SetDestructor(pycapsule, None)


class NpHolder:

    def __init__(self, np_array: np.ndarray) -> None:
        self.np_array = np_array
        self.data = np_array.ctypes.data_as(ctypes.c_void_p)
        self.shape = np_array.ctypes.shape_as(ctypes.c_int64)
        self.strides = np_array.ctypes.strides_as(ctypes.c_int64)
        for i in range(np_array.ndim):
            self.strides[i] //= np_array.itemsize


def make_manager_ctx(holder: NpHolder, dl_managed_tensor: dlpack.DLManagedTensor):
    py_obj = ctypes.py_object(holder)
    libmain_Debug(py_obj, ctypes.py_object(NpHolder))
    ctypes.pythonapi.Py_IncRef(py_obj)
    py_obj_ptr = ctypes.pointer(py_obj)
    ctypes.pythonapi.Py_IncRef(ctypes.py_object(py_obj_ptr))
    dl_managed_tensor.manager_ctx = ctypes.cast(py_obj_ptr, ctypes.c_void_p)


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

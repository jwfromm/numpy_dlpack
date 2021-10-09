import ctypes
import numpy as np
from .dlpack import _c_str_dltensor, DLManagedTensor

class _Holder:
    def __init__(self, array_interface, pycapsule) -> None:
        self.__array_interface__ = array_interface
        self._pycapsule = pycapsule


def to_numpy(array) -> np.ndarray:
    pycapsule = array.__dlpack__()
    pycapsule = ctypes.py_object(pycapsule)
    assert ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _c_str_dltensor)
    dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(pycapsule, _c_str_dltensor)
    dl_managed_tensor = ctypes.cast(dl_managed_tensor, ctypes.POINTER(DLManagedTensor))
    holder = _Holder(dl_managed_tensor.contents.__array_interface__, pycapsule)
    return np.ctypeslib.as_array(holder)

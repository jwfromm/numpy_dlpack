import ctypes

PyObjPtr = ctypes.POINTER(ctypes.py_object)
PyMem_RawMalloc = ctypes.pythonapi.PyMem_RawMalloc
PyMem_RawMalloc.argtypes = [ctypes.c_size_t]
PyMem_RawMalloc.restype = ctypes.c_void_p



global_address = None
# global_copy = None

def _alloc_dl_managed_tensor():
    global global_copy, global_address
    address = PyMem_RawMalloc(64)
    global_address = address
    print("address =", hex(address))
    result = ctypes.c_int.from_address(address)
    # global_copy = result
    return result

def proc1():
    value = _alloc_dl_managed_tensor()
    value.value = 10
    print(value)
    del value


def main():
    proc1()
    global global_address
    result = ctypes.c_int.from_address(global_address)
    print(result)


main()

import tvm
import numpy as np
import ctypes

libmain = ctypes.cdll.LoadLibrary("./libmain.so")

class DLDevice(ctypes.Structure):
  _fields_ = [("device_type", ctypes.c_int),
              ("device_id", ctypes.c_int)]

class DLDataTypeCode(ctypes.c_uint8):
	kDLInt = 0
	kDLUInt = 1
	kDLFloat = 2
	kDLBfloat = 4

	def __str__(self):
		return {self.kDLInt : 'int', self.kDLUInt : 'uint', self.kDLFloat : 'float', self.kDLBfloat : 'bfloat'}[self.value]              

class DLDataType(ctypes.Structure):
  _fields_ = [("type_code", DLDataTypeCode),
              ("bits", ctypes.c_uint8),
              ("lanes", ctypes.c_uint16)]
  TYPE_MAP = {
    "bool": (1, 1, 1),
    "int32": (0, 32, 1),
    "int64": (0, 64, 1),
    "uint32": (1, 32, 1),
    "uint64": (1, 64, 1),
    "float32": (2, 32, 1),
    "float64": (2, 64, 1),
  }

class DLTensor(ctypes.Structure):
  _fields_ = [("data", ctypes.c_void_p),
              ("device", DLDevice),
              ("ndim", ctypes.c_int),
              ("dtype", DLDataType),
              ("shape", ctypes.POINTER(ctypes.c_int64)),
              ("strides", ctypes.POINTER(ctypes.c_int64)),
              ("byte_offset", ctypes.c_uint64)]

  @property
  def size(self):
    prod = 1
    for i in range(self.ndim):
      prod *= self.shape[i]
    return prod

  @property
  def itemsize(self):
    return self.dtype.lanes * self.dtype.bits // 8;
  
  @property
  def nbytes(self):
    return self.size * self.itemsize 
  
  @property
  def __array_interface__(self):
    shape = tuple(self.shape[dim] for dim in range(self.ndim))
    if self.strides:
      strides = tuple(self.strides[dim] * self.itemsize for dim in range(self.ndim))
    else:
      # Array is compact, make it numpy compatible.
      strides = []
      for i, s in enumerate(shape):
        cumulative = 1
        for e in range(i + 1, self.ndim):
          cumulative *= shape[e]
        strides.append(cumulative * self.itemsize)
      strides = tuple(strides)
    typestr = '|' + str(self.dtype.type_code)[0] + str(self.itemsize)
    return dict(version = 3, shape = shape, strides = strides, data = (self.data, True), offset = self.byte_offset, typestr = typestr)

def __str__(self):
  return 'dtype={dtype}, ndim={ndim}, shape={shape}, strides={strides}, byte_offset={byte_offset}'.format(dtype = self.dtype, ndim = self.ndim, shape = tuple(self.shape[i] for i in range(self.ndim)), strides = tuple(self.strides[i] for i in range(self.ndim)), byte_offset = self.byte_offset) 

class DLManagedTensor(ctypes.Structure):
  pass

DLManagedTensorHandle = ctypes.POINTER(DLManagedTensor)

DeleterFunc = ctypes.CFUNCTYPE(None, DLManagedTensorHandle)

DLManagedTensor._fields_ = [("dl_tensor", DLTensor),
                            ("manager_ctx", ctypes.c_void_p),
                            ("deleter", DeleterFunc)]

def make_manager_ctx(obj):
  pyobj = ctypes.py_object(obj)
  void_p = ctypes.c_void_p.from_buffer(pyobj)
  ctypes.pythonapi.Py_IncRef(pyobj)
  return void_p

# N.B.: In practice, one should ensure that this function
# is not destructed before the numpy array is destructed.
@DeleterFunc
def dl_managed_tensor_deleter(dl_managed_tensor_handle):
  void_p = dl_managed_tensor_handle.contents.manager_ctx
  pyobj = ctypes.cast(void_p, ctypes.py_object)
  ctypes.pythonapi.Py_DecRef(pyobj)
  #libmain.FreeHandle()

PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
PyCapsule_New = ctypes.pythonapi.PyCapsule_New
PyCapsule_New.restype = ctypes.py_object
PyCapsule_New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.restype = ctypes.c_void_p
PyCapsule_GetPointer.argtypes = (ctypes.py_object, ctypes.c_char_p)

def make_capsule(dl_managed_tensor):
  return PyCapsule_New(ctypes.byref(dl_managed_tensor), b'dltensor', None)

def np_to_dlpack(array):
  # You may check array.flags here, e.g. array.flags['C_CONTIGUOUS']
  dl_tensor = DLTensor()
  dl_tensor.data = array.ctypes.data_as(ctypes.c_void_p)
  dl_tensor.device = DLDevice(1, 0)
  dl_tensor.ndim = array.ndim
  dl_tensor.dtype = DLDataType.TYPE_MAP[str(array.dtype)]
  # For 0-dim ndarrays, strides and shape will be NULL
  dl_tensor.shape = array.ctypes.shape_as(ctypes.c_int64)
  dl_tensor.strides = array.ctypes.strides_as(ctypes.c_int64)
  for i in range(array.ndim):
    dl_tensor.strides[i] //= array.itemsize
  dl_tensor.byte_offset = 0
  dl_managed_tensor= DLManagedTensor()
  dl_managed_tensor.dl_tensor = dl_tensor
  dl_managed_tensor.manager_ctx = make_manager_ctx(array)
  dl_managed_tensor.deleter = dl_managed_tensor_deleter
  return dl_managed_tensor

def dlpack_to_np(capsule):
  dl_managed_tensor = ctypes.cast(PyCapsule_GetPointer(capsule, b'dltensor'), ctypes.POINTER(DLManagedTensor)).contents
  wrapped = type('', (), dict(__array_interface__ = dl_managed_tensor.dl_tensor.__array_interface__))()
  return np.asarray(wrapped)

def dlpack_to_nd(capsule):
  return tvm.nd.from_dlpack(capsule)

def np_to_nd(array):
  dl_managed_tensor = np_to_dlpack(array)
  capsule = make_capsule(dl_managed_tensor)
  nd_array = dlpack_to_nd(capsule)
  libmain.Give(dl_managed_tensor)
  return nd_array

def nd_to_np(array):
  capsule = array.to_dlpack()
  np_array = dlpack_to_np(capsule)
  return np_array
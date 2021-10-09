#include <stdio.h>

#include "Python.h"
#include "dlpack.h"

int Debug(PyObject* obj, PyObject* cls) {
  int isNpHolder = PyObject_IsInstance(obj, cls);
  PyObject* s = PyObject_Repr(obj);
  PyObject* ss = PyUnicode_AsEncodedString(s, "utf-8", "~E~");
  const char* bytes = PyBytes_AS_STRING(ss);
  printf("[%s:%d] py_object(%p, refcnt=%ld, isNpHolder=%d): %s\n", __FILE__,
         __LINE__, obj, obj->ob_refcnt, isNpHolder, bytes);
  return 0;
}

int DebugDLManagedTensor(DLManagedTensor* dlt) {
  printf("DLManagedTensor @ %p\n", dlt);
  printf("  data = %p\n", dlt->dl_tensor.data);
  printf("  shape = [");
  for (int i = 0; i < dlt->dl_tensor.ndim; ++i) {
    if (i != 0) {
      printf(", ");
    }
    printf("%ld", dlt->dl_tensor.shape[i]);
  }
  printf("] @ %p\n", dlt->dl_tensor.shape);
  printf("  strides = [");
  if (dlt->dl_tensor.strides) {
    for (int i = 0; i < dlt->dl_tensor.ndim; ++i) {
      if (i != 0) {
        printf(", ");
      }
      printf("%ld", dlt->dl_tensor.strides[i]);
    }
    printf("] @ %p\n", dlt->dl_tensor.strides);
  } else {
    printf("NULL\n");
  }
  printf("  manager_ctx = %p\n", dlt->manager_ctx);
  printf("  *manager_ctx = %p\n", *((PyObject**)dlt->manager_ctx));
  return 0;  //
}

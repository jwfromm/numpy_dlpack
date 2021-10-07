#include <stdio.h>
#include <stdlib.h>
#include "dlpack.h"

DLManagedTensor *given = NULL;

void Give(DLManagedTensor dl_managed_tensor) {
  given = (DLManagedTensor *) malloc(sizeof(DLManagedTensor));
  *given = dl_managed_tensor;
}

void Finalize() {
  given->deleter(given);
}

void FreeHandle() {
  free(given);
  given = NULL;
}

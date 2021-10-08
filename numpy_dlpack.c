#include <stdlib.h>

#include "dlpack.h"

void *AllocDLManagedTensor() { return malloc(sizeof(DLManagedTensor)); }
void FreeDLManagedTensor(DLManagedTensor *p) { free(p); }


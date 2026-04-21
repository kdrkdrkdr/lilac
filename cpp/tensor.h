#ifndef LILAC_TENSOR_H
#define LILAC_TENSOR_H

#include <stdint.h>

#define LILAC_MAX_NDIM 8

typedef struct {
    const char *name;
    float      *data;
    uint32_t    ndim;
    uint32_t    shape[LILAC_MAX_NDIM];
    uint64_t    num_elems;
} Tensor;

/* Tagged struct so forward decls (`struct TensorStore`) match this definition. */
typedef struct TensorStore {
    Tensor  *tensors;
    uint32_t num_tensors;
    void    *blob;   /* aligned fp32 data buffer backing all tensors */
} TensorStore;

/* Load weights.bin. Returns 0 on success. */
int  tensor_store_load(TensorStore *store, const char *path);

void tensor_store_free(TensorStore *store);

/* Lookup by exact name. Returns NULL if not found. */
const Tensor *tensor_get(const TensorStore *store, const char *name);

#endif

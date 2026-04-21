#define _CRT_SECURE_NO_WARNINGS
#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(size, align) _aligned_malloc((size), (align))
#define ALIGNED_FREE(ptr)          _aligned_free((ptr))
#else
#include <stdlib.h>
static void *_aa(size_t size, size_t align) {
    void *p = NULL;
    if (posix_memalign(&p, align, size) != 0) return NULL;
    return p;
}
#define ALIGNED_ALLOC(size, align) _aa((size), (align))
#define ALIGNED_FREE(ptr)          free((ptr))
#endif

#define MAGIC   "LILC"
#define VERSION 1
#define ALIGN   32

static int read_exact(FILE *f, void *dst, size_t n) {
    return fread(dst, 1, n, f) == n ? 0 : -1;
}

int tensor_store_load(TensorStore *store, const char *path) {
    memset(store, 0, sizeof(*store));

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "tensor_store_load: cannot open %s\n", path);
        return -1;
    }

    char     magic[4];
    uint32_t version;
    uint32_t num_tensors;
    uint64_t data_size;

    if (read_exact(f, magic, 4) || memcmp(magic, MAGIC, 4) != 0) {
        fprintf(stderr, "tensor_store_load: bad magic\n");
        fclose(f);
        return -1;
    }
    if (read_exact(f, &version, 4) || version != VERSION) {
        fprintf(stderr, "tensor_store_load: bad version %u (want %d)\n", version, VERSION);
        fclose(f);
        return -1;
    }
    if (read_exact(f, &num_tensors, 4) || read_exact(f, &data_size, 8)) {
        fprintf(stderr, "tensor_store_load: bad header\n");
        fclose(f);
        return -1;
    }

    Tensor *tensors = (Tensor *)calloc(num_tensors, sizeof(Tensor));
    if (!tensors) {
        fclose(f);
        return -1;
    }

    /* Read tensor table. Stash data_offset into data pointer temporarily;
       we patch it to a real address after loading the data section. */
    for (uint32_t i = 0; i < num_tensors; i++) {
        Tensor *t = &tensors[i];
        uint32_t name_len;
        if (read_exact(f, &name_len, 4)) goto fail;
        char *name = (char *)malloc((size_t)name_len + 1);
        if (!name) goto fail;
        if (read_exact(f, name, name_len)) { free(name); goto fail; }
        name[name_len] = '\0';
        t->name = name;

        if (read_exact(f, &t->ndim, 4) || t->ndim > LILAC_MAX_NDIM) goto fail;

        t->num_elems = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (read_exact(f, &t->shape[d], 4)) goto fail;
            t->num_elems *= t->shape[d];
        }

        uint64_t data_offset, data_size_bytes;
        if (read_exact(f, &data_offset, 8) || read_exact(f, &data_size_bytes, 8)) goto fail;
        t->data = (float *)(uintptr_t)data_offset;
        (void)data_size_bytes;  /* derivable from num_elems */
    }

    void *blob = ALIGNED_ALLOC((size_t)data_size, ALIGN);
    if (!blob) goto fail;

    if (read_exact(f, blob, (size_t)data_size)) {
        ALIGNED_FREE(blob);
        goto fail;
    }

    for (uint32_t i = 0; i < num_tensors; i++) {
        uintptr_t off = (uintptr_t)tensors[i].data;
        tensors[i].data = (float *)((uint8_t *)blob + off);
    }

    store->tensors     = tensors;
    store->num_tensors = num_tensors;
    store->blob        = blob;
    fclose(f);
    return 0;

fail:
    for (uint32_t i = 0; i < num_tensors; i++) free((void *)tensors[i].name);
    free(tensors);
    fclose(f);
    return -1;
}

void tensor_store_free(TensorStore *store) {
    if (!store->tensors) return;
    for (uint32_t i = 0; i < store->num_tensors; i++) {
        free((void *)store->tensors[i].name);
    }
    free(store->tensors);
    if (store->blob) ALIGNED_FREE(store->blob);
    memset(store, 0, sizeof(*store));
}

const Tensor *tensor_get(const TensorStore *store, const char *name) {
    for (uint32_t i = 0; i < store->num_tensors; i++) {
        if (strcmp(store->tensors[i].name, name) == 0) return &store->tensors[i];
    }
    return NULL;
}

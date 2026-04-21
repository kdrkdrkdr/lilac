/* Sanity check for tensor loader: load weights.bin, print a few known
   tensors, compare first/last values with Python-dumped blob via manual
   inspection. */
#include "tensor.h"

#include <stdio.h>
#include <string.h>

static void print_tensor(const TensorStore *store, const char *name) {
    const Tensor *t = tensor_get(store, name);
    if (!t) { printf("  MISSING: %s\n", name); return; }
    printf("  %-48s ndim=%u shape=[", t->name, t->ndim);
    for (uint32_t d = 0; d < t->ndim; d++) printf("%u%s", t->shape[d], d + 1 < t->ndim ? "," : "");
    printf("] n=%llu first=%.6f last=%.6f\n",
           (unsigned long long)t->num_elems, t->data[0], t->data[t->num_elems - 1]);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "cpp/weights.bin";
    TensorStore store;
    if (tensor_store_load(&store, path) != 0) return 1;
    printf("loaded %u tensors from %s\n", store.num_tensors, path);

    /* Group counts by top-level prefix. */
    uint32_t dec = 0, enc_q = 0, flow = 0, ref_enc = 0;
    for (uint32_t i = 0; i < store.num_tensors; i++) {
        const char *n = store.tensors[i].name;
        if      (strncmp(n, "dec.",     4) == 0) dec++;
        else if (strncmp(n, "enc_q.",   6) == 0) enc_q++;
        else if (strncmp(n, "flow.",    5) == 0) flow++;
        else if (strncmp(n, "ref_enc.", 8) == 0) ref_enc++;
    }
    printf("  dec=%u enc_q=%u flow=%u ref_enc=%u  (expect 157/70/88/20)\n",
           dec, enc_q, flow, ref_enc);

    printf("\nsample tensors:\n");
    print_tensor(&store, "dec.conv_pre.weight");
    print_tensor(&store, "dec.conv_pre.bias");
    print_tensor(&store, "dec.conv_post.weight");
    print_tensor(&store, "dec.ups.0.weight");
    print_tensor(&store, "enc_q.pre.weight");
    print_tensor(&store, "enc_q.enc.in_layers.0.weight");
    print_tensor(&store, "flow.flows.0.pre.weight");
    print_tensor(&store, "ref_enc.gru.weight_ih_l0");
    print_tensor(&store, "ref_enc.proj.weight");

    tensor_store_free(&store);
    return 0;
}

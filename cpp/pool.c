#include "pool.h"

#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef struct {
    HANDLE  thread;
    HANDLE  start;   /* auto-reset — set by submit, waited by worker */
    HANDLE  done;    /* auto-reset — set by worker after task, waited by pool_wait */
    PoolFn  fn;
    void   *arg;
    volatile LONG pending; /* 1 if submitted and not yet waited on */
    volatile LONG stop;
} Worker;

struct Pool {
    int     n;
    Worker *w;
};

static DWORD WINAPI worker_proc(LPVOID param) {
    Worker *w = (Worker *)param;
    for (;;) {
        WaitForSingleObject(w->start, INFINITE);
        if (w->stop) break;
        if (w->fn) w->fn(w->arg);
        SetEvent(w->done);
    }
    return 0;
}

Pool *pool_create(int n) {
    if (n <= 0) return NULL;
    Pool *p = (Pool *)calloc(1, sizeof(Pool));
    p->n = n;
    p->w = (Worker *)calloc((size_t)n, sizeof(Worker));
    for (int i = 0; i < n; i++) {
        p->w[i].start = CreateEventW(NULL, FALSE, FALSE, NULL);
        p->w[i].done  = CreateEventW(NULL, FALSE, FALSE, NULL);
        p->w[i].thread = CreateThread(NULL, 0, worker_proc, &p->w[i], 0, NULL);
    }
    return p;
}

void pool_destroy(Pool *p) {
    if (!p) return;
    for (int i = 0; i < p->n; i++) {
        p->w[i].stop = 1;
        SetEvent(p->w[i].start);
        WaitForSingleObject(p->w[i].thread, INFINITE);
        CloseHandle(p->w[i].thread);
        CloseHandle(p->w[i].start);
        CloseHandle(p->w[i].done);
    }
    free(p->w);
    free(p);
}

void pool_submit(Pool *p, int i, PoolFn fn, void *arg) {
    Worker *w = &p->w[i];
    w->fn = fn; w->arg = arg;
    w->pending = 1;
    SetEvent(w->start);
}

void pool_wait(Pool *p) {
    for (int i = 0; i < p->n; i++) {
        if (InterlockedExchange(&p->w[i].pending, 0) == 1) {
            WaitForSingleObject(p->w[i].done, INFINITE);
        }
    }
}

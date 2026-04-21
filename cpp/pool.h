#ifndef LILAC_POOL_H
#define LILAC_POOL_H

#ifdef __cplusplus
extern "C" {
#endif

/* Minimal fixed-size worker pool for dec's 3-way resblock parallelism. The
   caller (main thread) executes one task itself and dispatches the other two
   to workers — so a pool of 2 workers gives 3-way parallelism total. */
typedef struct Pool Pool;

typedef void (*PoolFn)(void *arg);

Pool *pool_create(int n_workers);
void  pool_destroy(Pool *p);

/* Submit task index (0 <= i < n_workers). Non-blocking. */
void  pool_submit(Pool *p, int i, PoolFn fn, void *arg);

/* Block until all submitted tasks complete. */
void  pool_wait(Pool *p);

#ifdef __cplusplus
}
#endif

#endif

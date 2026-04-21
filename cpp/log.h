#ifndef LILAC_LOG_H
#define LILAC_LOG_H

#ifdef __cplusplus
extern "C" {
#endif

/* Diagnostic logging. Writes to OutputDebugString + a `lilac.log` file next
   to the module. Safe to call from any thread. */
void lilac_log(const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif

#include "log.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/* Log sink: OutputDebugString (visible in DebugView) + a `lilac.log` file next
   to whichever module owns this function (DLL when loaded by Electron, EXE
   for console tests). Using a log file works around Electron's GUI subsystem
   not plumbing native stdout to any tty. */
void lilac_log(const char *fmt, ...) {
    char buf[512];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    OutputDebugStringA(buf);

    static FILE *lf = NULL;
    static int  tried = 0;
    if (!lf && !tried) {
        tried = 1;
        char path[MAX_PATH];
        HMODULE h = NULL;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                               | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR)&lilac_log, &h)
            && GetModuleFileNameA(h, path, MAX_PATH)) {
            char *slash = strrchr(path, '\\');
            if (slash) strcpy(slash + 1, "lilac.log");
            else       strcpy(path, "lilac.log");
            lf = fopen(path, "w");
        }
    }
    if (lf) { fputs(buf, lf); fflush(lf); }
}

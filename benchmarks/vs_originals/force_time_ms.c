/* LD_PRELOAD shim for timing FORCE's stages (force_tsa_timing.sh).
 *
 * FORCE's higher-level progress (progress-hl.c, utils-cl.c: proctime) times the input, compute and output stages with
 * time(), whole seconds, and prints them as "Virtual I-time / C-time / O-time" in the end summary. Too coarse for a
 * compute stage of a second or less. This replaces time() with a monotonic clock in milliseconds, so every "second"
 * FORCE prints is a millisecond (the summary's y/m/d/h/m/s fields are decoded back by the timing script). FORCE itself
 * is not touched. It only uses time() to measure these stages.
 *
 *     gcc -O2 -shared -fPIC -o force_time_ms.so force_time_ms.c
 */
#define _GNU_SOURCE
#include <time.h>

time_t time(time_t *t) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    time_t ms = (time_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    if (t) *t = ms;
    return ms;
}

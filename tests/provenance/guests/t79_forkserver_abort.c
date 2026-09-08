/* t48: forkserver consecutive-child-timeout abort.  The guest detects a
 * UAF and then loops forever, so every binradar-mode iteration hits the
 * forkserver child-timeout cap.  After
 * BINRADAR_FORKSERVER_TIMEOUT_ABORT_COUNT consecutive timeouts the
 * mutation plan must abort (remaining = 0, "[forkserver] [abort]
 * [consecutive-timeout N]") instead of burning one child timeout per
 * remaining patch id. */
#include <stdlib.h>

int main(void) {
    char *p = malloc(8);
    if (!p) return 0;
    free(p);
    for (;;) {
        p[0] = 1; /* UAF on every iteration; never returns */
    }
    return 0;
}
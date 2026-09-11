/* t80: pre-entry XMM memory access under semantic-event instrumentation.
 *
 * Regression for the binradar-phase tracer abort where an OSPREY/memcheck
 * semantic-event helper call is emitted between a guest `movss xmm, ea`
 * load and its XMM register store, breaking the symbolic engine's
 * load/store pairing (see
 * problem/TRACER_XMM_STORE_ABORT_OSPREY_SEM_EVENTS.md).
 *
 * The constructor runs BEFORE main, i.e. before the forkserver entrypoint
 * TB, and performs an explicit `movss` memory -> XMM -> memory round trip
 * plus an XMM register copy.  With semantic events active the load's XMM
 * store is separated from its producer by the injected helper call.
 *
 * Before the fix the tracer aborts while translating the constructor, so
 * the forkserver banner is never written and the driver sees a banner EOF.
 * After the fix the handshake succeeds, the child runs the constructor and
 * main, and exits normally.
 *
 * No memory error is expected: the finding must stay empty. */
#include <stdlib.h>

static float g_src = 1.5f;
static float g_dst;

/* movss memory -> xmm0 -> memory.  The explicit operands force the
 * translator's SSE 0x210 (movss xmm, ea) and 0x211 (movss ea, xmm)
 * forms, i.e. the load->XMM-store pair that the semantic-event helper
 * call separates.
 *
 * The reg-reg `movss %xmm0,%xmm1` form is deliberately NOT exercised: it
 * lowers to a 32-bit XMM lane load/store pair that the symbolic engine
 * has never modelled and still aborts on (a pre-existing limitation,
 * unrelated to semantic events; see the problem doc). */
__attribute__((constructor, noinline, used))
static void preentry_movss(void) {
    __asm__ volatile("movss %1, %%xmm0\n\t"
                     "movss %%xmm0, %0\n\t"
                     : "=m"(g_dst)
                     : "m"(g_src)
                     : "xmm0", "memory");
}

int main(void) {
    char *p = malloc(32);
    if (!p) return 0;
    free(p);
    return 0;
}

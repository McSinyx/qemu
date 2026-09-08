/*
 * OSPREY Stage 7.1 and 7.3 focused tests: baseline access identity and
 * immutable mutation-plan ownership.
 *
 * Three layers:
 *
 *  - Locator capture matrix (drives the OSPREY runtime catalogs
 *    directly through the public hooks and checks
 *    osprey_capture_address_ref/osprey_capture_chunk_ref identity
 *    contracts: global/heap/stack cell locators, recursive stack
 *    activations, two live same-site heaps, numeric heap-base reuse
 *    across provenance generations, pointer target locator, NULL
 *    target, stale/freed rejection, and the interval rejection
 *    matrix).  Order-sensitive identity cases run under two actual
 *    insertion orders and check the distinctness rules that would fail
 *    if identity were reduced to raw address, canonical region alone,
 *    or site alone.
 *
 *  - Record-level groups: this translation unit includes
 *    linux-user/snapshot.c, which makes the child-side record writers
 *    and the parent-side analyze_collected_data() reachable (their
 *    state is static in snapshot.c).  snapshot.c also provides
 *    log_msg/trace_mem/is_valid_address/mr_manager_heap_search_pub, so
 *    the stage3_test_support.c stubs are NOT linked here (duplicate
 *    strong definitions).  The record groups cover: record replacement
 *    at one raw cell with a different width, pointer-over-primitive
 *    replacement with moved-array back-reference repair, exact-at-cap
 *    records and first-over-cap sticky flag handling, and parent
 *    sort/traversal using only the validated count.
 *
 *  - Owned-plan groups: exact legacy generic descriptors, untyped-pointer
 *    fallback, independent fresh payloads, allocation rollback, private
 *    child application, FIFO transitions, and total manager teardown.
 *
 * Build: see tests/osprey/Makefile (targets unit-stage7-mutation and
 * unit-stage7-mutation-asan).
 */

#include "qemu/osdep.h"
#include "osprey.h"
#include "osprey-internal.h"
#include "snapshot.h"
#include "provenance.h"
#include "qemu/thread.h"
#include "tcg/symbolic/symbolic-struct.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

/* snapshot.c compiled in: full access to its statics. */
#include "../../linux-user/snapshot.c"

/* ------------------------------------------------------------------ */
/* Linker environment (externs snapshot.c and the OSPREY objects read) */
/* ------------------------------------------------------------------ */

unsigned long guest_base = 0;
uint64_t symbolic_start_code = 0;
uint64_t symbolic_end_code = 0;
Expr *pool = NULL;
Expr *next_free_expr = NULL;
Query *query_queue = NULL;
Query *next_query = NULL;
__thread CPUState *thread_cpu = NULL;
abi_ulong mmap_next_start = 0;
target_ulong target_brk = 0;
uint64_t last_translation_block = 0;

/* Host-environment stubs for snapshot.o's integration points.  The
 * record-level groups exercise the access-record paths only, not page
 * management or forkserver transport. */
void memcheck_init(void) {}
abi_long target_mmap(abi_ulong start, abi_ulong len, int prot, int flags,
                     int fd, abi_ulong offset) {
    (void)start; (void)len; (void)prot; (void)flags; (void)fd; (void)offset;
    return (abi_long)-1;
}
int page_get_flags(target_ulong address) { (void)address; return 0; }
int walk_memory_regions(void *priv, walk_memory_regions_fn fn) {
    (void)priv; (void)fn;
    return 0;
}
void rcu_disable_atfork(void) {}

/* No-op mutex stubs: the unit runner is single-threaded. */
void qemu_mutex_init(QemuMutex *m) { memset(m, 0, sizeof(*m)); }
void qemu_mutex_destroy(QemuMutex *m) { (void)m; }
void qemu_mutex_lock_impl(QemuMutex *m, const char *f, const int l) {
    (void)m; (void)f; (void)l;
}
void qemu_mutex_unlock_impl(QemuMutex *m, const char *f, const int l) {
    (void)m; (void)f; (void)l;
}
QemuMutexLockFunc qemu_mutex_lock_func = qemu_mutex_lock_impl;
QemuMutexTrylockFunc qemu_mutex_trylock_func = NULL;

/* ------------------------------------------------------------------ */
/* Test scaffolding                                                    */
/* ------------------------------------------------------------------ */

static unsigned failures;
static unsigned checks;

#define CHECK(_condition, _message) do {                                  \
    checks++;                                                              \
    if (!(_condition)) {                                                   \
        fprintf(stderr, "FAIL: %s (line %d)\n", (_message), __LINE__); \
        failures++;                                                        \
    }                                                                      \
} while (0)

/* Reinitialize the runtime catalogs so identity cases start from a
 * known state. */
static void reset_runtime(void)
{
    osprey_free_runtime_regions();
    provenance_init();
    osprey_set_image_bounds(0x400000, 0x401000);
    osprey_register_image_global(NULL, 0x402000, 0x1000); /* .data */
    osprey_register_image_global(NULL, 0x403000, 0x800);  /* .bss */
}

/* Register a live heap instance the same way the allocator hook does:
 * provenance object first, then osprey_on_alloc_success. */
static PtrTag make_heap(CPUArchState *env, target_ulong base,
                        target_ulong size, target_ulong site_pc)
{
    PtrTag t = provenance_create_object(base, size, site_pc,
                                        PROV_PRODUCER_MALLOC_RETURN);
    OspreyAllocatorObservation obs = {
        .kind = OSPREY_ALLOCATOR_MALLOC,
        .site_pc = site_pc,
        .requested_size = size,
    };
    osprey_on_alloc_success(env, &obs, base, t.object_id, t.generation);
    return t;
}

static void free_heap(CPUArchState *env, const PtrTag *t, target_ulong site_pc)
{
    osprey_on_free_identity(env, t->object_id, t->generation, site_pc);
    CHECK(provenance_retire_object(t->concrete_value),
          "provenance object retired by base");
}

/* ------------------------------------------------------------------ */
/* Capture matrix                                                      */
/* ------------------------------------------------------------------ */

/* Group 1: global, heap, and stack cell locators (both orders). */
static void test_cell_locators(void)
{
    reset_runtime();
    CPUArchState *env = g_malloc0(sizeof(CPUArchState));

    /* Global: merged instance 0, image-relative offset. */
    OspreyRuntimeAddressRef gref;
    CHECK(osprey_capture_address_ref(env, 0x402008, &gref),
          "global capture");
    CHECK(gref.valid == 1 && gref.instance_id == 0 &&
          gref.prov_object_id == 0 && gref.prov_generation == 0,
          "global locator identity");
    CHECK(gref.address.region.kind == OSPREY_REGION_GLOBAL &&
          gref.address.offset == 0x2008 && gref.raw == 0x402008,
          "global canonical roundtrip");

    /* Heap: distinct instance + provenance pair. */
    PtrTag t = make_heap(env, 0x5000, 64, 0x400200);
    OspreyRuntimeAddressRef href;
    CHECK(osprey_capture_address_ref(env, 0x5010, &href),
          "heap capture");
    CHECK(href.valid == 1 && href.instance_id != 0,
          "heap instance id nonzero");
    CHECK(href.prov_object_id == t.object_id &&
          href.prov_generation == t.generation,
          "heap provenance pair");
    CHECK(href.address.region.kind == OSPREY_REGION_HEAP_SITE &&
          href.address.offset == 0x10 && href.raw == 0x5010,
          "heap canonical roundtrip");
    CHECK(href.address.region.site_offset == 0x200,
          "heap site normalized");

    /* Stack: two nested precise frames; the innermost wins with signed
     * offsets relative to its own entry SP. */
    osprey_on_call(env, 0x400300, 0x7ff000); /* outer frame */
    osprey_on_call(env, 0x400400, 0x7fef00); /* callee frame */
    osprey_on_rsp_update(env, 0x7feeec, 0x400410); /* callee grows */
    OspreyRuntimeAddressRef sref;
    CHECK(osprey_capture_address_ref(env, 0x7feeec, &sref),
          "stack capture");
    CHECK(sref.valid == 1 && sref.instance_id != 0,
          "stack instance id nonzero");
    CHECK(sref.address.region.kind == OSPREY_REGION_STACK_FUNCTION,
          "stack region kind");
    CHECK(sref.address.region.site_offset == 0x400,
          "callee site normalized");
    CHECK(sref.address.offset == (int64_t)0x7feeec - (int64_t)0x7fef00,
          "stack signed offset vs entry SP");

    /* Caller-frame address resolves to the outer frame, not the
     * callee: distinct instance id under the same canonical region
     * kind.  The caller address sits inside the caller's red-zone
     * window but above the callee entry SP, so the innermost-first
     * scan skips the callee. */
    OspreyRuntimeAddressRef cref;
    CHECK(osprey_capture_address_ref(env, 0x7fefb0, &cref),
          "caller-frame capture");
    CHECK(cref.valid == 1 && cref.instance_id != sref.instance_id,
          "caller and callee frames distinct instances");
    CHECK(cref.address.region.site_offset == 0x300,
          "caller site normalized");
    CHECK(cref.address.offset == (int64_t)0x7fefb0 - (int64_t)0x7ff000,
          "caller signed offset vs entry SP");

    /* Every captured locator round-trips raw<->canonical through the
     * instance anchor. */
    CHECK((target_ulong)((int64_t)(target_ulong)gref.raw -
                         gref.address.offset) == 0x402008 - 0x2008,
          "global anchor reconstructs raw");
    CHECK((target_ulong)((int64_t)href.raw - href.address.offset) == 0x5000,
          "heap anchor reconstructs raw");
    CHECK((target_ulong)((int64_t)sref.raw - sref.address.offset) ==
              0x7fef00,
          "stack anchor reconstructs raw");

    free_heap(env, &t, 0x400400);
    g_free(env);
}

/* Group 2: recursive same-site activations must never collapse. */
static void test_recursive_frames(void)
{
    reset_runtime();
    CPUArchState *env = g_malloc0(sizeof(CPUArchState));

    osprey_on_call(env, 0x400300, 0x7ff000);
    OspreyRuntimeAddressRef r1;
    CHECK(osprey_capture_address_ref(env, 0x7feff8, &r1),
          "outer recursive frame capture");

    /* Recursion: same callee PC, lower entry SP. */
    osprey_on_call(env, 0x400300, 0x7fef00);
    osprey_on_rsp_update(env, 0x7feef0, 0x400310);
    OspreyRuntimeAddressRef r2;
    CHECK(osprey_capture_address_ref(env, 0x7feef8, &r2),
          "inner recursive frame capture");

    CHECK(r1.valid == 1 && r2.valid == 1, "both recursive frames");
    CHECK(r1.instance_id != r2.instance_id,
          "recursive activations distinct instance ids");
    CHECK(r1.address.region.kind == r2.address.region.kind &&
          r1.address.region.site_offset == r2.address.region.site_offset,
          "same canonical region (site)");
    CHECK(r1.address.offset == r2.address.offset,
          "same entry-relative offset");
    /* Region+offset alone cannot distinguish the two: the locator must
     * carry instance_id. */
    CHECK(r1.raw != r2.raw, "recursive raw addresses differ");

    g_free(env);
}

/* Group 3: two live same-site heap allocations. */
static void test_two_live_heaps(int order)
{
    reset_runtime();
    CPUArchState *env = g_malloc0(sizeof(CPUArchState));

    PtrTag t1, t2;
    if (order == 0) {
        t1 = make_heap(env, 0x6000, 32, 0x400200);
        t2 = make_heap(env, 0x7000, 32, 0x400200);
    } else {
        t2 = make_heap(env, 0x7000, 32, 0x400200);
        t1 = make_heap(env, 0x6000, 32, 0x400200);
    }

    OspreyRuntimeAddressRef a1, a2;
    if (order == 0) {
        CHECK(osprey_capture_address_ref(env, 0x6008, &a1),
              "heap 1 capture (order 0)");
        CHECK(osprey_capture_address_ref(env, 0x7008, &a2),
              "heap 2 capture (order 0)");
    } else {
        CHECK(osprey_capture_address_ref(env, 0x7008, &a2),
              "heap 2 capture (order 1)");
        CHECK(osprey_capture_address_ref(env, 0x6008, &a1),
              "heap 1 capture (order 1)");
    }

    CHECK(a1.valid == 1 && a2.valid == 1, "both live heaps resolve");
    CHECK(a1.instance_id != a2.instance_id,
          "same-site live heaps distinct instance ids");
    CHECK(a1.address.region.kind == a2.address.region.kind &&
          a1.address.region.site_offset == a2.address.region.site_offset,
          "same site, so region identity collides");
    CHECK(a1.prov_object_id == t1.object_id &&
          a2.prov_object_id == t2.object_id,
          "provenance pair separates them");
    CHECK(a1.raw != a2.raw, "raw addresses differ");

    free_heap(env, &t1, 0x400400);
    free_heap(env, &t2, 0x400400);
    g_free(env);
}

/* Group 4: numeric heap-base reuse across provenance generations. */
static void test_base_reuse(void)
{
    reset_runtime();
    CPUArchState *env = g_malloc0(sizeof(CPUArchState));

    PtrTag old = make_heap(env, 0x8000, 16, 0x400200);
    OspreyRuntimeAddressRef ref_old;
    CHECK(osprey_capture_address_ref(env, 0x8008, &ref_old),
          "old generation capture");
    uint64_t old_instance = ref_old.instance_id;
    CHECK(ref_old.prov_object_id == old.object_id, "old prov pair");

    free_heap(env, &old, 0x400400);
    PtrTag cur = make_heap(env, 0x8000, 64, 0x400200);
    OspreyRuntimeAddressRef ref_new;
    CHECK(osprey_capture_address_ref(env, 0x8008, &ref_new),
          "new generation capture");

    CHECK(ref_new.instance_id != old_instance,
          "reused base gets a fresh instance id");
    CHECK(ref_new.prov_object_id != old.object_id,
          "reused base gets a fresh provenance object");
    CHECK(ref_new.address.region.site_offset ==
              ref_old.address.region.site_offset,
          "same allocation site");

    free_heap(env, &cur, 0x400400);
    g_free(env);
}

/* A pair that remains LIVE in the object table is still stale when the
 * authoritative live-by-base index names a newer allocation. */
static void test_live_base_authority(void)
{
    reset_runtime();
    CPUArchState *env = g_malloc0(sizeof(CPUArchState));
    PtrTag old = make_heap(env, 0x8800, 32, 0x400200);
    PtrTag newer = provenance_create_object(0x8800, 32, 0x400200,
                                             PROV_PRODUCER_MALLOC_RETURN);

    OspreyRuntimeAddressRef ref;
    CHECK(!osprey_capture_address_ref(env, 0x8808, &ref),
          "superseded live-by-base identity rejected");
    CHECK(ref.valid == 0, "superseded capture clears output");

    osprey_on_free_identity(env, old.object_id, old.generation, 0x400400);
    CHECK(provenance_retire_object(newer.concrete_value),
          "new authoritative provenance object retired");
    ProvenanceObject *old_object = provenance_lookup_object(
        old.object_id, old.generation);
    CHECK(old_object != NULL, "historical provenance object retained");
    if (old_object != NULL) {
        old_object->state = PROV_OBJ_FREED;
    }
    g_free(env);
}

/* Group 5 + 6: pointer target locator semantics. */
static void test_pointer_target_locator(int order)
{
    reset_runtime();
    CPUArchState *env = g_malloc0(sizeof(CPUArchState));

    PtrTag t;
    /* Exercise both catalog insertion orders: the cell lives on the
     * stack and the target is the heap object. */
    if (order == 0) {
        t = make_heap(env, 0x9000, 64, 0x400200);
        osprey_on_call(env, 0x400300, 0x7ff000);
    } else {
        osprey_on_call(env, 0x400300, 0x7ff000);
        t = make_heap(env, 0x9000, 64, 0x400200);
    }

    OspreyRuntimeChunkRef cell;
    CHECK(osprey_capture_chunk_ref(env, 0x7feff8,
                                   (target_ulong)sizeof(target_ulong),
                                   &cell),
          order ? "pointer cell chunk (order 1)"
                : "pointer cell chunk (order 0)");
    CHECK(cell.start.valid == 1 && cell.size == sizeof(target_ulong),
          "cell locator width");

    /* Non-NULL target: address locator from the live target instance. */
    OspreyRuntimeAddressRef tref;
    CHECK(osprey_capture_address_ref(env, 0x9020, &tref),
          "target capture");
    CHECK(tref.prov_object_id == t.object_id && tref.instance_id != 0,
          "target locator from live instance");
    CHECK(tref.raw == 0x9020, "target raw address");

    /* NULL target: capture of 0 must fail (no fabricated locator). */
    OspreyRuntimeAddressRef nref;
    CHECK(!osprey_capture_address_ref(env, 0, &nref),
          "NULL target not capturable");
    CHECK(nref.valid == 0, "failed capture leaves zeroed locator");

    free_heap(env, &t, 0x400400);
    g_free(env);
}

/* Group 7: stale/freed heap target rejection. */
static void test_stale_target(void)
{
    reset_runtime();
    CPUArchState *env = g_malloc0(sizeof(CPUArchState));

    PtrTag t = make_heap(env, 0xa000, 32, 0x400200);
    free_heap(env, &t, 0x400400);

    OspreyRuntimeAddressRef ref;
    CHECK(!osprey_capture_address_ref(env, 0xa008, &ref),
          "freed heap rejected");
    CHECK(ref.valid == 0, "freed capture leaves zeroed locator");

    g_free(env);
}

/* Disabled capture is still a failed capture and must clear caller-owned
 * output so a reused record cannot retain a stale valid locator. */
static void test_disabled_capture_clears_output(void)
{
    OspreyRuntimeAddressRef address;
    OspreyRuntimeChunkRef chunk;
    memset(&address, 0xa5, sizeof(address));
    memset(&chunk, 0xa5, sizeof(chunk));

    osprey_collect_enabled = 0;
    CHECK(!osprey_capture_address_ref(NULL, 0x402008, &address),
          "disabled address capture rejected");
    CHECK(!osprey_capture_chunk_ref(NULL, 0x402008, 8, &chunk),
          "disabled chunk capture rejected");
    osprey_collect_enabled = 1;

    OspreyRuntimeAddressRef zero_address;
    OspreyRuntimeChunkRef zero_chunk;
    memset(&zero_address, 0, sizeof(zero_address));
    memset(&zero_chunk, 0, sizeof(zero_chunk));
    CHECK(memcmp(&address, &zero_address, sizeof(address)) == 0,
          "disabled address capture clears output");
    CHECK(memcmp(&chunk, &zero_chunk, sizeof(chunk)) == 0,
          "disabled chunk capture clears output");
}

/* Group 8: interval rejection matrix. */
static void test_interval_rejection(void)
{
    reset_runtime();
    CPUArchState *env = g_malloc0(sizeof(CPUArchState));
    PtrTag t = make_heap(env, 0xb000, 32, 0x400200);

    OspreyRuntimeChunkRef chunk;

    /* Zero width. */
    CHECK(!osprey_capture_chunk_ref(env, 0xb000, 0, &chunk),
          "zero width rejected");

    /* End overflow. */
    CHECK(!osprey_capture_chunk_ref(env, (target_ulong)-1, 2, &chunk),
          "end overflow rejected");

    /* Full-object interval valid; half-open end boundary. */
    CHECK(osprey_capture_chunk_ref(env, 0xb000, 32, &chunk),
          "full-object interval valid");
    CHECK(chunk.start.address.offset == 0 && chunk.size == 32,
          "full-object locator fields");
    CHECK(!osprey_capture_chunk_ref(env, 0xb018, 9, &chunk),
          "interval ending past object rejected");
    CHECK(!osprey_capture_chunk_ref(env, 0xb020, 1, &chunk),
          "one-past-end byte rejected");

    /* One-byte crossing into an unmapped neighbor: same as
     * one-past-end for a single byte. */
    CHECK(!osprey_capture_chunk_ref(env, 0xb020, 1, &chunk),
          "crossing out of instance rejected");

    free_heap(env, &t, 0x400400);
    g_free(env);
}

/* Cross-instance interval: both endpoints valid but different
 * instances -> rejected (identity reduction check). */
static void test_interval_cross_instance(int order)
{
    reset_runtime();
    CPUArchState *env = g_malloc0(sizeof(CPUArchState));

    /* Two adjacent heaps; an interval spanning both must be rejected. */
    PtrTag t1, t2;
    if (order == 0) {
        t1 = make_heap(env, 0xc000, 8, 0x400200);
        t2 = make_heap(env, 0xc008, 8, 0x400200);
    } else {
        t2 = make_heap(env, 0xc008, 8, 0x400200);
        t1 = make_heap(env, 0xc000, 8, 0x400200);
    }

    OspreyRuntimeChunkRef chunk;
    CHECK(!osprey_capture_chunk_ref(env, 0xc000, 16, &chunk),
          order ? "cross-instance interval rejected (order 1)"
                : "cross-instance interval rejected (order 0)");

    /* Reduction-to-site check: the two heaps share the site; if the
     * chunk locator carried only region identity it would accept this
     * span.  Individual endpoints remain distinct instances. */
    OspreyRuntimeAddressRef r1, r2;
    bool resolved = order == 0
        ? osprey_capture_address_ref(env, 0xc000, &r1) &&
          osprey_capture_address_ref(env, 0xc008, &r2)
        : osprey_capture_address_ref(env, 0xc008, &r2) &&
          osprey_capture_address_ref(env, 0xc000, &r1);
    CHECK(resolved, "adjacent heaps individually resolvable");
    CHECK(r1.instance_id != r2.instance_id, "adjacent heaps distinct");

    free_heap(env, &t1, 0x400400);
    free_heap(env, &t2, 0x400400);
    g_free(env);
}

static void test_capture_matrix(void)
{
    test_disabled_capture_clears_output();
    test_live_base_authority();
    test_cell_locators();
    test_recursive_frames();
    test_base_reuse();
    test_stale_target();
    test_interval_rejection();
    for (int order = 0; order < 2; order++) {
        test_two_live_heaps(order);
        test_pointer_target_locator(order);
        test_interval_cross_instance(order);
    }
}

/* ------------------------------------------------------------------ */
/* Record-level groups (snapshot.c compiled in)                        */
/* ------------------------------------------------------------------ */

/* Simulated guest memory: the parent's analysis path dereferences
 * record addresses through g2h(), and the pointer path validates
 * targets against g_snapshot.pages.  One host allocation backs the
 * low guest addresses used by the record tests; record addresses must
 * stay inside [TEST_GUEST_BASE, TEST_GUEST_BASE + TEST_GUEST_SPAN). */
#define TEST_GUEST_SPAN (2u * 1024u * 1024u)
#define TEST_GUEST_BASE 0x10000000u
static void *test_guest_mem = NULL;

static void setup_guest_memory(void)
{
    test_guest_mem = g_malloc0(TEST_GUEST_SPAN);
    guest_base = (unsigned long)test_guest_mem - TEST_GUEST_BASE;
    /* Register the pages as snapshot-writable so is_valid_address(x,
     * true) accepts pointer targets. */
    for (target_ulong page = 0; page < TEST_GUEST_SPAN;
         page += SNAPSHOT_PAGE_SIZE) {
        SnapshotPageInfo *info = g_malloc0(sizeof(SnapshotPageInfo));
        info->addr = TEST_GUEST_BASE + page;
        info->perms = PAGE_READ | PAGE_WRITE;
        info->data = (void *)((unsigned long)test_guest_mem + page);
        g_hash_table_insert(g_snapshot.pages,
                            GSIZE_TO_POINTER(TEST_GUEST_BASE + page), info);
    }
}

static void teardown_guest_memory(void)
{
    guest_base = 0;
    if (test_guest_mem != NULL) {
        g_free(test_guest_mem);
        test_guest_mem = NULL;
    }
    if (g_snapshot.pages != NULL) {
        /* The pages table has a NULL value destroy: free the infos. */
        GHashTableIter it;
        gpointer key, value;
        g_hash_table_iter_init(&it, g_snapshot.pages);
        while (g_hash_table_iter_next(&it, &key, &value)) {
            g_free(value);
        }
        g_hash_table_remove_all(g_snapshot.pages);
    }
}

/* Drive snapshot_read_access the way the symbolic helpers do: a
 * SnapshotMemAccess with the loaded value copied into .target. */
static void record_access(CPUArchState *env, uintptr_t addr,
                          target_ulong value, int size, bool symbolic)
{
    SnapshotMemAccess ma;
    memset(&ma, 0, sizeof(ma));
    ma.addr = addr;
    ma.size = (uintptr_t)size;
    ma.symbolic_value = symbolic;
    ma.pc = 0x400100;
    memcpy(ma.target, &value, sizeof(target_ulong));
    snapshot_read_access(env, &ma);
}

/* Reset the child-side record state (statics of snapshot.c are directly
 * reachable in this translation unit). */
static void reset_shared_records(void)
{
    if (shared_trace_data != NULL) {
        memset(shared_trace_data, 0, sizeof(SharedTraceData));
    }
    /* There is no ordered_map_destroy in snapshot.c; free the entries
     * (the table owns them) and the containers. */
    if (g_read_access_pointers != NULL) {
        g_hash_table_destroy(g_read_access_pointers->table);
        g_queue_free(g_read_access_pointers->queue);
        g_free(g_read_access_pointers);
        g_read_access_pointers = NULL;
    }
    if (g_read_access_tainted_primitives != NULL) {
        g_hash_table_destroy(g_read_access_tainted_primitives->table);
        g_queue_free(g_read_access_tainted_primitives->queue);
        g_free(g_read_access_tainted_primitives);
        g_read_access_tainted_primitives = NULL;
    }
    snapshot_modification_manager_reset(false);
}

/* Group 9: record replacement at one raw cell with a different width.
 * The heap catalog lives inside the simulated guest span so both the
 * capture path and the parent's g2h() dereference resolve. */
static void test_record_width_replacement(void)
{
    reset_runtime();
    CPUArchState *env = g_malloc0(sizeof(CPUArchState));
    PtrTag t = make_heap(env, TEST_GUEST_BASE + 0x5000, 64, 0x400200);
    reset_shared_records();

    uintptr_t cell = TEST_GUEST_BASE + 0x5010;
    record_access(env, cell, 0, 8, true);
    CHECK(shared_trace_data->prim_idx == 1, "8-byte primitive recorded");
    CHECK(shared_trace_data->primitives[0].cell.start.valid == 1 &&
          shared_trace_data->primitives[0].cell.size == 8,
          "8-byte cell locator captured");
    CHECK(shared_trace_data->primitives[0].cell.start.address.offset == 0x10,
          "cell canonical offset");

    /* Same raw cell, 4-byte symbolic access: replacement updates both
     * width and locator consistently. */
    record_access(env, cell, 0, 4, true);
    CHECK(shared_trace_data->prim_idx == 1, "one record retained");
    CHECK(shared_trace_data->primitives[0].size == 4, "width replaced");
    CHECK(shared_trace_data->primitives[0].cell.size == 4,
          "locator width replaced consistently");

    /* Stale-locator check: retire the heap identity, then re-record at
     * the same cell.  The new record must carry NO stale locator. */
    free_heap(env, &t, 0x400400);
    record_access(env, cell, 0, 8, true);
    CHECK(shared_trace_data->primitives[0].cell.start.valid == 0,
          "stale identity leaves zeroed locator");

    reset_shared_records();
    g_free(env);
}

/* Group 10: pointer over primitive at one cell, then a later primitive
 * write at the same cell exercises the swap-removal path and the
 * moved-array back-reference repair.  The pointer writer itself does
 * NOT remove primitive records; the removal happens when a subsequent
 * primitive access finds a pointer entry at the aligned cell. */
static void test_pointer_over_primitive(void)
{
    reset_runtime();
    CPUArchState *env = g_malloc0(sizeof(CPUArchState));
    PtrTag target_heap = make_heap(env, TEST_GUEST_BASE + 0x6000, 64,
                                   0x400200);
    /* A second heap hosts the pointer cell so the cell locator has a
     * live instance of its own (distinct from the target instance). */
    PtrTag cell_heap = make_heap(env, TEST_GUEST_BASE + 0x6800, 64,
                                 0x400300);
    reset_shared_records();

    uintptr_t cell = TEST_GUEST_BASE + 0x6820;
    target_ulong target = TEST_GUEST_BASE + 0x6020;

    /* 1. Primitive at a cell. */
    record_access(env, cell, 0, 8, true);
    CHECK(shared_trace_data->prim_idx == 1, "primitive first");

    /* 2. Pointer access at the same cell: records a pointer.  The
     * primitive record stays until the next primitive write finds the
     * pointer entry (removal happens on the primitive path). */
    record_access(env, cell, target, 8, false);
    CHECK(shared_trace_data->ptr_idx == 1, "pointer recorded");
    CHECK(shared_trace_data->pointers[0].cell.start.valid == 1 &&
              shared_trace_data->pointers[0].cell.start.prov_object_id ==
                  cell_heap.object_id,
          "pointer cell locator from cell instance");
    CHECK(shared_trace_data->pointers[0].target_ref.valid == 1 &&
          shared_trace_data->pointers[0].target_ref.prov_object_id ==
              target_heap.object_id,
          "target locator identity");
    CHECK(shared_trace_data->pointers[0].target_ref.raw == target,
          "target locator raw");
    CHECK(shared_trace_data->pointers[0].cell.start.instance_id !=
              shared_trace_data->pointers[0].target_ref.instance_id,
          "cell and target instances distinct");

    /* 3. Add five primitives at distinct cells (fills the array and
     * sets up the swap-removal move). */
    for (uintptr_t a = 0; a < 40; a += 8) {
        record_access(env, TEST_GUEST_BASE + 0x7100 + a, 0, 8, true);
    }
    CHECK(shared_trace_data->prim_idx == 6,
          "primitives recorded (stale one not yet removed)");

    /* 4. Primitive write at the pointer cell: the pointer entry
     * suppresses it and removes the stale primitive via swap-removal
     * (moves the last primitive into its slot, repairs the moved
     * back-reference). */
    record_access(env, cell, 0, 8, true);
    CHECK(shared_trace_data->prim_idx == 5,
          "primitive at pointer cell removed by swap-removal");

    /* All pointer records keep consistent locators after the moves. */
    for (uint32_t i = 0; i < shared_trace_data->ptr_idx; i++) {
        PointerAccess *p = &shared_trace_data->pointers[i];
        if (p->cell.start.valid) {
            CHECK(p->cell.size == 8, "pointer locator width");
            CHECK(p->target_ref.valid == 1 &&
                      p->target_ref.prov_object_id ==
                          target_heap.object_id,
                  "pointer target locator");
        }
    }
    /* No primitive record retained at the pointer cell. */
    for (uint32_t i = 0; i < shared_trace_data->prim_idx; i++) {
        PrimitiveAccess *p = &shared_trace_data->primitives[i];
        CHECK(p->addr != cell, "no stale primitive at pointer cell");
    }

    free_heap(env, &target_heap, 0x400400);
    free_heap(env, &cell_heap, 0x400400);
    reset_shared_records();
    g_free(env);
}

/* Group 11: exact-at-cap records, LRU recycling, and the defensive
 * over-cap path.  The defensive branch calls exit_with_status(1), so
 * it runs in a forked child; the sticky flag lives in MAP_SHARED
 * memory and survives into the parent. */
static void test_at_cap_sticky(void)
{
    reset_runtime();
    CPUArchState *env = g_malloc0(sizeof(CPUArchState));
    reset_shared_records();

    /* Fill exactly to capacity. */
    for (int i = 0; i < MAX_PRIMITIVE_ACCESS; i++) {
        record_access(env, (uintptr_t)(TEST_GUEST_BASE + 0x8000 +
                                       8 * (uintptr_t)i), 0, 8, true);
    }
    CHECK(shared_trace_data->prim_idx == MAX_PRIMITIVE_ACCESS,
          "exactly at cap");
    CHECK(shared_trace_data->prim_overflow == 0, "no overflow at cap");

    /* One more distinct cell: LRU recycles the oldest shared_index, so
     * the counter stays at cap and nothing overflows. */
    record_access(env, (uintptr_t)(TEST_GUEST_BASE + 0x8000 - 8), 0, 8,
                  true);
    CHECK(shared_trace_data->prim_idx == MAX_PRIMITIVE_ACCESS,
          "index bounded after LRU recycling");
    CHECK(shared_trace_data->prim_overflow == 0,
          "LRU recycling does not set the sticky flag");

    /* Start a fresh record set for the defensive path: an ordered-map
     * entry whose shared_index points past the array (== MAX) trips
     * the writer's out-of-range branch, which sets the sticky flag and
     * exits.  Run it in a forked child while the queue is BELOW cap:
     * an LRU pop at cap would overwrite the injected shared_index
     * before the writer sees it. */
    reset_shared_records();
    CHECK(g_read_access_tainted_primitives == NULL,
          "record set cleared before defensive-path fork");
    pid_t pid = fork();
    CHECK(pid >= 0, "fork for defensive path");
    if (pid == 0) {
        if (g_read_access_tainted_primitives == NULL) {
            g_read_access_tainted_primitives =
                ordered_map_init(MAX_PRIMITIVE_ACCESS);
        }
        OrderedMapEntry *entry = g_new(OrderedMapEntry, 1);
        memset(entry, 0, sizeof(*entry));
        entry->key = TEST_GUEST_BASE + 0x9000;
        entry->shared_index = MAX_PRIMITIVE_ACCESS; /* out of range */
        g_hash_table_insert(g_read_access_tainted_primitives->table,
                            GSIZE_TO_POINTER(entry->key), entry);
        g_queue_push_tail(g_read_access_tainted_primitives->queue, entry);
        entry->node = g_read_access_tainted_primitives->queue->tail;
        record_access(env, entry->key, 0, 8, true);
        _exit(0); /* not reached: exit_with_status(1) fires first */
    }
    int wstatus = 0;
    CHECK(waitpid(pid, &wstatus, 0) == pid, "child reaped");
    CHECK(WIFEXITED(wstatus) && WEXITSTATUS(wstatus) == 1,
          "defensive path exits without a signal");
    CHECK(shared_trace_data->prim_overflow == 1,
          "defensive path sets sticky flag");

    /* A corrupted UINT32_MAX count must not become a negative signed
     * index and write before the fixed array. */
    reset_shared_records();
    pid = fork();
    CHECK(pid >= 0, "fork for extreme count");
    if (pid == 0) {
        shared_trace_data->prim_idx = UINT32_MAX;
        record_access(env, TEST_GUEST_BASE + 0x9010, 0, 8, true);
        _exit(0);
    }
    wstatus = 0;
    CHECK(waitpid(pid, &wstatus, 0) == pid, "extreme-count child reaped");
    CHECK(WIFEXITED(wstatus) && WEXITSTATUS(wstatus) == 1,
          "extreme count exits without out-of-bounds write");
    CHECK(shared_trace_data->prim_idx == UINT32_MAX,
          "writer does not advance corrupt count");
    CHECK(shared_trace_data->prim_overflow == 1,
          "extreme count sets sticky flag");

    reset_shared_records();
    pid = fork();
    CHECK(pid >= 0, "fork for extreme pointer count");
    if (pid == 0) {
        shared_trace_data->ptr_idx = UINT32_MAX;
        record_access(env, TEST_GUEST_BASE + 0x9018,
                      TEST_GUEST_BASE + 0x100, 8, false);
        _exit(0);
    }
    wstatus = 0;
    CHECK(waitpid(pid, &wstatus, 0) == pid,
          "extreme-pointer-count child reaped");
    CHECK(WIFEXITED(wstatus) && WEXITSTATUS(wstatus) == 1,
          "extreme pointer count exits without out-of-bounds write");
    CHECK(shared_trace_data->ptr_idx == UINT32_MAX,
          "pointer writer does not advance corrupt count");
    CHECK(shared_trace_data->ptr_overflow == 1,
          "extreme pointer count sets sticky flag");

    /* Removal from a corrupt empty record set must not underflow the
     * published count. */
    reset_shared_records();
    g_read_access_tainted_primitives =
        ordered_map_init(MAX_PRIMITIVE_ACCESS);
    OrderedMapEntry *entry = ordered_map_insert(
        g_read_access_tainted_primitives, TEST_GUEST_BASE + 0x9020, NULL);
    entry->shared_index = 0;
    remove_read_access_primitive(entry->key);
    CHECK(shared_trace_data->prim_idx == 0,
          "corrupt removal does not underflow count");
    CHECK(shared_trace_data->prim_overflow == 1,
          "corrupt removal sets sticky flag");

    reset_shared_records();
    g_free(env);
}

/* A valid bounded record whose locator is unavailable must retain the
 * existing generic mutation descriptors and order. */
static void test_generic_without_locator(void)
{
    reset_runtime();
    CPUArchState *env = g_malloc0(sizeof(CPUArchState));
    reset_shared_records();

    uintptr_t cell = TEST_GUEST_BASE + 0x9800;
    osprey_collect_enabled = 0;
    record_access(env, cell, 0, 8, true);
    osprey_collect_enabled = 1;
    CHECK(shared_trace_data->prim_idx == 1,
          "generic-only record retained");
    CHECK(shared_trace_data->primitives[0].cell.start.valid == 0,
          "generic-only record has no locator");
    shared_trace_data->exit_info.valid = 1;

    ArgumentInfo arg_info[1] = {{0, "rax", 0, NULL}};
    int remaining = analyze_collected_data(arg_info, 0);
    CHECK(mod_manager != NULL && mod_manager->current != NULL,
          "generic-only record selects a candidate");
    if (mod_manager != NULL && mod_manager->current != NULL) {
        SnapshotMutationWrite *candidate = &mod_manager->current->mods[0];
        CHECK(candidate->addr == cell && candidate->size == 8 &&
              candidate->kind == SNAPSHOT_MUTATION_BYTES,
              "generic-only descriptor unchanged");
        CHECK(g_queue_get_length(mod_manager->modifications) == 1 &&
              remaining == 2,
              "generic-only candidate order/count unchanged");
    }

    reset_shared_records();
    g_free(env);
}

/* Group 12: parent sort/traversal uses only a validated count.  An
 * over-cap count does not identify an initialized prefix, so the parent
 * must reject that array entirely rather than sort or synthesize generic
 * candidates from zeroed/unpublished slots. */
static void test_parent_count_clamp(void)
{
    reset_runtime();
    CPUArchState *env = g_malloc0(sizeof(CPUArchState));
    reset_shared_records();

    uintptr_t base = TEST_GUEST_BASE + 0xa000;
    record_access(env, base, 0, 8, true);
    record_access(env, base + 8, 0, 8, true);
    record_access(env, base + 0x10, 0, 8, true);
    CHECK(shared_trace_data->prim_idx == 3, "three records");

    /* Simulate a corrupted counter plus the sticky overflow flag.  The
     * three real records are no longer distinguishable from an invented
     * prefix, so the parent must consume none of this array. */
    shared_trace_data->prim_idx = MAX_PRIMITIVE_ACCESS + 7;
    shared_trace_data->prim_overflow = 1;
    shared_trace_data->exit_info.valid = 1;

    ArgumentInfo arg_info[1] = {{0, "rax", 0, NULL}};
    int remaining = analyze_collected_data(arg_info, 0);

    CHECK(mod_manager == NULL && mutation_analysis_started,
          "empty corrupt input destroys the exhausted manager");
    CHECK(remaining == 0, "corrupt count reports no remaining work");
    CHECK(g_hash_table_size(g_read_access_tainted_primitives_all) == 0,
          "corrupt count traverses no primitive records");

    reset_shared_records();
    g_free(env);
}

/* ------------------------------------------------------------------ */
/* Stage 7.3 owned-plan groups                                         */
/* ------------------------------------------------------------------ */

static void free_plan_queue(GQueue *queue)
{
    if (queue != NULL) {
        g_queue_free_full(queue, (GDestroyNotify)snapshot_mutation_free);
    }
}

static void test_owned_generic_plan_parity(void)
{
    static const uint32_t widths[] = {1, 2, 3, 4, 8};
    for (size_t wi = 0; wi < G_N_ELEMENTS(widths); wi++) {
        MutationCandidate source;
        memset(&source, 0x5a, sizeof(source));
        source.addr = wi == 0 ? 3 : TEST_GUEST_BASE + 0xa800 + wi * 8;
        source.size = widths[wi];
        source.kind = 0;
        source.expr = (Expr *)(uintptr_t)0x1234;
        MutationCandidate before = source;
        GQueue *queue = g_queue_new();

        snapshot_mutation_test_set_alloc_fail_after(-1);
        bool queued = add_modification_primitive(queue, &source);
        CHECK(queued, "generic plan batch queued");
        CHECK(memcmp(&source, &before, sizeof(source)) == 0,
              "generic planning leaves source unchanged");
        CHECK(g_queue_get_length(queue) == (widths[wi] == 3 ? 1 : 3),
              "generic plan count matches legacy order");

        uint32_t ordinal = 0;
        for (GList *node = queue->head; node != NULL; node = node->next) {
            Modification *modification = node->data;
            CHECK(modification->num_mods == 1,
                  "generic queue entries have one owned write");
            SnapshotMutationWrite *write = &modification->mods[0];
            CHECK(write->kind == SNAPSHOT_MUTATION_BYTES &&
                  write->addr == source.addr && write->size == source.size &&
                  write->expr == source.expr && write->target.bytes == NULL &&
                  write->target.extent == 0,
                  "generic plan descriptor is private and inactive-target clean");
            if (source.size == 3) {
                uint8_t expected[sizeof(source.value)];
                memcpy(expected, source.value, sizeof(expected));
                flip_bits(expected, source.size);
                CHECK(memcmp(write->value, expected, sizeof(expected)) == 0,
                      "default-width plan preserves bit-flip bytes");
            } else if (ordinal == 0) {
                uint8_t expected[sizeof(source.value)];
                memcpy(expected, source.value, sizeof(expected));
                memset(expected, 0, source.size);
                CHECK(memcmp(write->value, expected, sizeof(expected)) == 0,
                      "zero plan bytes match source width");
            } else if (ordinal == 1) {
                uint8_t expected[sizeof(source.value)];
                memcpy(expected, source.value, sizeof(expected));
                memset(expected, 0, source.size);
                expected[0] = 1;
                CHECK(memcmp(write->value, expected, sizeof(expected)) == 0,
                      "one plan bytes match source width");
            } else {
                uint8_t expected[sizeof(source.value)];
                memcpy(expected, source.value, sizeof(expected));
                /* Preserve the legacy descriptors, not the intended
                 * algorithm: the old one-byte path flipped a zeroed
                 * working value, and the old two-byte path flipped the
                 * value after replacing it with one. */
                if (source.size == 1) {
                    expected[0] = 0;
                } else if (source.size == 2) {
                    expected[0] = 1;
                    expected[1] = 0;
                }
                flip_bits(expected, source.size);
                CHECK(memcmp(write->value, expected, sizeof(expected)) == 0,
                      "bit-flip plan preserves legacy descriptor");
            }
            ordinal++;
        }
        free_plan_queue(queue);
    }
}

static void test_owned_generic_boundary_parity(void)
{
    static const struct {
        uint32_t size;
        uint64_t original;
        uint32_t count;
        uint64_t expected[3];
    } cases[] = {
        {1, 0, 2, {1, UINT8_MAX}},
        {1, 1, 2, {0, UINT8_MAX}},
        {2, 0, 2, {1, UINT16_MAX - 1}},
        {2, 1, 2, {0, UINT16_MAX - 1}},
        {3, 0x030201, 1, {0xfcfdfe}},
        {4, 0, 2, {1, UINT32_MAX}},
        {4, 1, 2, {0, UINT32_MAX - 1}},
        {8, 0, 2, {1, UINT64_MAX}},
        {8, 1, 2, {0, UINT64_MAX - 1}},
    };

    for (size_t ci = 0; ci < G_N_ELEMENTS(cases); ci++) {
        MutationCandidate source = {0};
        source.addr = TEST_GUEST_BASE + 0xaa00 + ci * 8;
        source.size = cases[ci].size;
        source.expr = (Expr *)(uintptr_t)(0x2000 + ci);
        memcpy(source.value, &cases[ci].original, source.size);
        MutationCandidate before = source;
        GQueue *queue = g_queue_new();

        CHECK(add_modification_primitive(queue, &source),
              "boundary generic batch queued");
        CHECK(memcmp(&source, &before, sizeof(source)) == 0,
              "boundary generic source remains immutable");
        CHECK(g_queue_get_length(queue) == cases[ci].count,
              "boundary generic plan count matches legacy");

        uint32_t ordinal = 0;
        for (GList *node = queue->head; node != NULL; node = node->next) {
            Modification *modification = node->data;
            uint8_t expected[sizeof(target_ulong)] = {0};
            memcpy(expected, &cases[ci].expected[ordinal], source.size);
            CHECK(modification->num_mods == 1 &&
                  modification->mods[0].kind == SNAPSHOT_MUTATION_BYTES &&
                  modification->mods[0].size == source.size &&
                  memcmp(modification->mods[0].value, expected,
                         source.size) == 0,
                  "boundary generic descriptor matches legacy oracle");
            ordinal++;
        }
        free_plan_queue(queue);
    }
}

static void test_owned_untyped_pointer_parity(void)
{
    MutationCandidate source = {0};
    source.addr = TEST_GUEST_BASE + 0xaac0;
    source.size = sizeof(target_ulong);
    source.kind = 1;
    source.expr = (Expr *)(uintptr_t)0x4567;
    target_ulong original = TEST_GUEST_BASE + 0x100;
    memcpy(source.value, &original, sizeof(original));
    MutationCandidate before = source;
    GQueue *queue = g_queue_new();

    CHECK(add_untyped_pointer_candidate(queue, &source),
          "untyped pointer plan queued");
    CHECK(memcmp(&source, &before, sizeof(source)) == 0,
          "untyped pointer source remains immutable");
    CHECK(g_queue_get_length(queue) == 1,
          "untyped pointer keeps one legacy null plan");
    if (!g_queue_is_empty(queue)) {
        Modification *modification = g_queue_peek_head(queue);
        target_ulong value = UINT64_MAX;
        memcpy(&value, modification->mods[0].value, sizeof(value));
        CHECK(modification->num_mods == 1 &&
              modification->mods[0].kind == SNAPSHOT_MUTATION_POINTER_NULL &&
              modification->mods[0].addr == source.addr &&
              modification->mods[0].size == sizeof(target_ulong) &&
              modification->mods[0].expr == source.expr && value == 0 &&
              modification->mods[0].target.bytes == NULL &&
              modification->mods[0].target.extent == 0,
              "untyped pointer descriptor preserves null fallback");
    }
    free_plan_queue(queue);

    queue = g_queue_new();
    snapshot_mutation_test_set_alloc_fail_after(0);
    CHECK(!add_untyped_pointer_candidate(queue, &source),
          "untyped pointer allocation failure is reported");
    snapshot_mutation_test_set_alloc_fail_after(-1);
    CHECK(g_queue_is_empty(queue) &&
          memcmp(&source, &before, sizeof(source)) == 0,
          "untyped pointer failure leaves queue and source unchanged");
    free_plan_queue(queue);
}

static void test_owned_fresh_payloads(void)
{
    MutationCandidate source;
    memset(&source, 0, sizeof(source));
    source.addr = TEST_GUEST_BASE + 0xab00;
    source.size = sizeof(target_ulong);
    source.kind = 1;
    source.expr = (Expr *)(uintptr_t)0x5678;
    uint8_t zero[sizeof(target_ulong)] = {0};
    uint8_t ones[sizeof(target_ulong)];
    memset(ones, 1, sizeof(ones));
    Modification *batch[2] = {
        snapshot_mutation_new(&source, SNAPSHOT_MUTATION_POINTER_FRESH,
                              zero, source.size, sizeof(zero), zero),
        snapshot_mutation_new(&source, SNAPSHOT_MUTATION_POINTER_FRESH,
                              zero, source.size, sizeof(ones), ones),
    };
    GQueue *queue = g_queue_new();
    CHECK(batch[0] != NULL && batch[1] != NULL,
          "fresh plans allocate independently");
    CHECK(snapshot_mutation_enqueue_batch(queue, batch, 2),
          "fresh plan batch enqueues atomically");
    if (g_queue_get_length(queue) == 2) {
        Modification *first = g_queue_peek_nth(queue, 0);
        Modification *second = g_queue_peek_nth(queue, 1);
        CHECK(first->mods[0].target.bytes != second->mods[0].target.bytes,
              "fresh target payloads do not alias");
        CHECK(memcmp(first->mods[0].target.bytes, zero, sizeof(zero)) == 0 &&
              memcmp(second->mods[0].target.bytes, ones, sizeof(ones)) == 0,
              "fresh target payloads retain independent fills");
        CHECK(first->mods[0].value[0] == 0 &&
              second->mods[0].value[0] == 0,
              "fresh cell bytes remain immutable placeholders");
    }
    free_plan_queue(queue);
}

static void test_atomic_plan_enqueue_failures(void)
{
    bool saw_success = false;
    MutationCandidate source;
    memset(&source, 0x5a, sizeof(source));
    source.addr = TEST_GUEST_BASE + 0xac00;
    source.size = sizeof(target_ulong);
    source.expr = (Expr *)(uintptr_t)0x9abc;

    for (int64_t fail_after = 0; fail_after < 64; fail_after++) {
        GQueue *queue = g_queue_new();
        snapshot_mutation_test_set_alloc_fail_after(-1);
        Modification *sentinel = snapshot_mutation_new(
            &source, SNAPSHOT_MUTATION_BYTES, source.value, source.size,
            0, NULL);
        CHECK(sentinel != NULL, "sentinel plan allocates");
        CHECK(snapshot_mutation_enqueue_one(queue, sentinel),
              "sentinel plan enqueues");
        MutationCandidate before = source;

        snapshot_mutation_test_set_alloc_fail_after(fail_after);
        bool queued = add_modification_primitive(queue, &source);
        snapshot_mutation_test_set_alloc_fail_after(-1);
        if (queued) {
            CHECK(g_queue_get_length(queue) == 4,
                  "successful batch appends all plans");
            CHECK(memcmp(&source, &before, sizeof(source)) == 0,
                  "successful batch leaves source unchanged");
            saw_success = true;
            free_plan_queue(queue);
            break;
        }
        CHECK(g_queue_get_length(queue) == 1,
              "failed batch leaves queue byte-count unchanged");
        CHECK(memcmp(&source, &before, sizeof(source)) == 0,
              "failed batch leaves source unchanged");
        free_plan_queue(queue);
    }
    CHECK(saw_success, "allocation-failure sweep reaches queue success");
}

static void test_nested_payload_failures(void)
{
    bool saw_success = false;
    MutationCandidate source;
    memset(&source, 0, sizeof(source));
    source.addr = TEST_GUEST_BASE + 0xac80;
    source.size = sizeof(target_ulong);
    uint8_t zero[sizeof(target_ulong)] = {0};
    uint8_t ones[sizeof(target_ulong)];
    memset(ones, 1, sizeof(ones));

    for (int64_t fail_after = 0; fail_after < 32; fail_after++) {
        GQueue *queue = g_queue_new();
        snapshot_mutation_test_set_alloc_fail_after(-1);
        Modification *sentinel = snapshot_mutation_new(
            &source, SNAPSHOT_MUTATION_BYTES, source.value, source.size,
            0, NULL);
        CHECK(sentinel != NULL, "nested-failure sentinel allocates");
        CHECK(snapshot_mutation_enqueue_one(queue, sentinel),
              "nested-failure sentinel enqueues");

        Modification *batch[2] = {NULL, NULL};
        snapshot_mutation_test_set_alloc_fail_after(fail_after);
        batch[0] = snapshot_mutation_new(
            &source, SNAPSHOT_MUTATION_POINTER_FRESH, zero, source.size,
            sizeof(zero), zero);
        batch[1] = snapshot_mutation_new(
            &source, SNAPSHOT_MUTATION_POINTER_FRESH, zero, source.size,
            sizeof(ones), ones);
        bool queued = false;
        if (batch[0] != NULL && batch[1] != NULL) {
            queued = snapshot_mutation_enqueue_batch(queue, batch, 2);
        } else {
            snapshot_mutation_free_batch(batch, 2);
        }
        snapshot_mutation_test_set_alloc_fail_after(-1);

        if (queued) {
            CHECK(g_queue_get_length(queue) == 3,
                  "nested-failure success appends both plans");
            saw_success = true;
            free_plan_queue(queue);
            break;
        }
        CHECK(g_queue_get_length(queue) == 1,
              "nested-failure batch leaves queue unchanged");
        free_plan_queue(queue);
    }
    CHECK(saw_success, "nested allocation-failure sweep reaches success");
}

static void test_owned_plan_validation(void)
{
    MutationCandidate source = {0};
    source.addr = CPU_NB_REGS;
    source.size = sizeof(target_ulong);
    CHECK(snapshot_mutation_new(&source, SNAPSHOT_MUTATION_BYTES,
                                source.value, source.size, 0, NULL) == NULL,
          "invalid register selector is rejected");

    source.addr = TEST_GUEST_BASE + 0xacc0;
    CHECK(snapshot_mutation_new(&source, (SnapshotMutationKind)-1,
                                source.value, source.size, 0, NULL) == NULL,
          "invalid mutation kind is rejected");
    CHECK(snapshot_mutation_new(&source, SNAPSHOT_MUTATION_POINTER_NULL,
                                source.value, 4, 0, NULL) == NULL,
          "non-native pointer width is rejected");
}

static void test_manager_fifo_and_cleanup(void)
{
    reset_shared_records();
    SnapshotExitInfo exit_info = {0};
    mod_manager_init(&exit_info);
    CHECK(mod_manager != NULL && mutation_analysis_started,
          "manager initialization owns analysis state");

    MutationCandidate source = {0};
    source.addr = TEST_GUEST_BASE + 0xace0;
    source.size = sizeof(target_ulong);
    Modification *batch[3] = {NULL, NULL, NULL};
    for (uint32_t i = 0; i < G_N_ELEMENTS(batch); i++) {
        source.value[0] = (uint8_t)(0x11 * (i + 1));
        batch[i] = snapshot_mutation_new(
            &source, SNAPSHOT_MUTATION_BYTES, source.value, source.size,
            0, NULL);
    }
    CHECK(snapshot_mutation_enqueue_batch(mod_manager->modifications,
                                          batch, G_N_ELEMENTS(batch)),
          "manager FIFO batch queued");

    for (uint32_t i = 0; i < 3; i++) {
        int remaining = select_next_modification(&exit_info);
        CHECK(mod_manager != NULL && mod_manager->current != NULL &&
              mod_manager->current->mods[0].value[0] ==
                  (uint8_t)(0x11 * (i + 1)),
              "manager selects plans in FIFO order");
        CHECK(remaining == (int)(3 - i),
              "manager reports current plus queued count");
    }
    CHECK(select_next_modification(&exit_info) == 0,
          "manager reports exhausted queue");
    CHECK(mod_manager == NULL && mutation_analysis_started,
          "exhaustion destroys manager without reopening analysis");
    CHECK(select_next_modification(&exit_info) == 0,
          "exhausted manager remains terminal");
    snapshot_modification_manager_reset(false);

    for (uint32_t iteration = 0; iteration < 32; iteration++) {
        mod_manager_init(&exit_info);
        source.value[0] = (uint8_t)iteration;
        mod_manager->current = snapshot_mutation_new(
            &source, SNAPSHOT_MUTATION_BYTES, source.value, source.size,
            0, NULL);
        Modification *queued = snapshot_mutation_new(
            &source, SNAPSHOT_MUTATION_BYTES, source.value, source.size,
            0, NULL);
        CHECK(mod_manager->current != NULL && queued != NULL &&
              snapshot_mutation_enqueue_one(mod_manager->modifications,
                                            queued),
              "manager cleanup cycle owns current and queued plans");
        snapshot_modification_manager_reset(false);
        CHECK(mod_manager == NULL && !mutation_analysis_started,
              "manager cleanup cycle releases all ownership");
    }
}

static void test_child_application_does_not_mutate_plan(void)
{
    reset_shared_records();
    CPUArchState *env = g_malloc0(sizeof(CPUArchState));
    MutationCandidate source;
    memset(&source, 0, sizeof(source));
    source.addr = TEST_GUEST_BASE + 0xad00;
    source.size = sizeof(target_ulong);
    source.expr = (Expr *)(uintptr_t)0xdef0;
    target_ulong value = 0x1122334455667788ULL;
    memcpy(source.value, &value, sizeof(value));

    Modification *plan = snapshot_mutation_new(
        &source, SNAPSHOT_MUTATION_BYTES, source.value, source.size, 0, NULL);
    CHECK(plan != NULL, "application plan allocates");
    if (plan != NULL) {
        uint8_t before[sizeof(plan->mods[0].value)];
        memcpy(before, plan->mods[0].value, sizeof(before));
        mod_manager = g_new0(ModificationManager, 1);
        mod_manager->modifications = g_queue_new();
        mod_manager->current = plan;
        snapshot_modify_memory(env);
        CHECK(memcmp(plan->mods[0].value, before, sizeof(before)) == 0,
              "child application preserves queued plan bytes");
        target_ulong applied = 0;
        memcpy(&applied, g2h(source.addr), sizeof(applied));
        CHECK(applied == value, "child application writes private value");
        snapshot_modification_manager_reset(false);
    }

    memset(&source, 0, sizeof(source));
    source.addr = R_EAX;
    source.size = sizeof(target_ulong);
    value = 0x8877665544332211ULL;
    memcpy(source.value, &value, sizeof(value));
    plan = snapshot_mutation_new(
        &source, SNAPSHOT_MUTATION_BYTES, source.value, source.size, 0, NULL);
    CHECK(plan != NULL, "register application plan allocates");
    if (plan != NULL) {
        uint8_t before[sizeof(plan->mods[0].value)];
        memcpy(before, plan->mods[0].value, sizeof(before));
        mod_manager = g_new0(ModificationManager, 1);
        mod_manager->modifications = g_queue_new();
        mod_manager->current = plan;
        mutation_analysis_started = true;
        snapshot_modify_memory(env);
        CHECK(env->regs[R_EAX] == value,
              "register destination uses the legacy selector contract");
        CHECK(memcmp(plan->mods[0].value, before, sizeof(before)) == 0,
              "register application preserves queued plan bytes");
        snapshot_modification_manager_reset(false);
    }
    g_free(env);
}

/* ------------------------------------------------------------------ */
/* Entry                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    /* The at-cap matrix records thousands of accesses; diagnostics are
     * not assertions and would obscure the focused result. */
    g_setenv("BINRADAR_TRACE_FILE", "none", TRUE);
    g_setenv("BINRADAR_TRACER_LOG_FILE", "none", TRUE);
    /* Arm the record machinery: forkserver_installed is a non-static
     * global in snapshot.c. */
    forkserver_installed = true;
    snapshot_init();
    CHECK(shared_trace_data != NULL, "shared trace data allocated");
    /* Deterministic OSPREY collection: the record writers gate capture
     * on this flag. */
    osprey_collect_enabled = 1;
    /* Back the guest addresses used by the record-level groups so
     * g2h() dereferences and pointer-target validation resolve. */
    setup_guest_memory();

    test_capture_matrix();

    test_record_width_replacement();
    test_pointer_over_primitive();
    test_at_cap_sticky();
    test_generic_without_locator();
    test_parent_count_clamp();
    test_owned_generic_plan_parity();
    test_owned_generic_boundary_parity();
    test_owned_untyped_pointer_parity();
    test_owned_fresh_payloads();
    test_atomic_plan_enqueue_failures();
    test_nested_payload_failures();
    test_owned_plan_validation();
    test_manager_fifo_and_cleanup();
    test_child_application_does_not_mutate_plan();

    osprey_free_runtime_regions();
    teardown_guest_memory();
    /* Free queued and current plans through the production destructor. */
    if (mod_manager != NULL) {
        snapshot_modification_manager_reset(false);
    }
    /* The analyze copies (prim_data/ptr_data) live in the original/all
     * hash tables with NULL value destroy.  The 'original' and 'all'
     * tables share the SAME copies, so free through the 'all' tables
     * only. */
    {
        GHashTable *tabs[] = {g_read_access_tainted_primitives_all,
                              g_read_access_pointers_all};
        GHashTable **aliases[] = {
            &g_read_access_tainted_primitives_original,
            &g_read_access_pointers_original};
        for (size_t t = 0; t < sizeof(tabs) / sizeof(tabs[0]); t++) {
            if (tabs[t] == NULL) continue;
            GHashTableIter it;
            gpointer key, value;
            g_hash_table_iter_init(&it, tabs[t]);
            while (g_hash_table_iter_next(&it, &key, &value)) {
                g_free(value);
            }
            g_hash_table_destroy(tabs[t]);
            if (t == 0) g_read_access_tainted_primitives_all = NULL;
            if (t == 1) g_read_access_pointers_all = NULL;
            /* The alias table references freed copies; drop it without
             * touching values. */
            if (*aliases[t] != NULL) {
                g_hash_table_destroy(*aliases[t]);
                *aliases[t] = NULL;
            }
        }
    }
    /* Drop the OSPREY context (frees the catalogs, relations, graph,
     * model, and per-CPU origin shadows) and the snapshot state. */
    if (g_osprey_ctx != NULL) {
        osprey_free(g_osprey_ctx);
        g_osprey_ctx = NULL;
    }
    if (g_snapshot.pages != NULL) {
        g_hash_table_destroy(g_snapshot.pages);
        g_snapshot.pages = NULL;
    }
    if (failures > 0) {
        fprintf(stderr, "stage7-mutation: %u/%u checks failed\n",
                failures, checks);
        return 1;
    }
    printf("stage7-mutation: %u checks passed\n", checks);
    return 0;
}
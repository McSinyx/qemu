/* t14_typed_mutation: Stage-7.5 applied-state fixture.
 *
 * Two live heap aggregates come from one allocation site; the selected
 * pointer names the second instance, so a first-instance resolver is wrong.
 * A second pointer is assigned only after the main-entry snapshot, proving
 * that parent-side planning uses the baseline child's recorded value rather
 * than stale snapshot memory.  The test-only child observation hook runs
 * before guest execution resumes after each mutation.
 */
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct T14Pair {
    uint64_t first;
    uint64_t second;
} T14Pair;

static T14Pair *t14_instances[2];
static T14Pair * volatile t14_pointer_cell;
static T14Pair * volatile t14_late_pointer_cell;
static T14Pair * volatile t14_null_cell;
static volatile uint64_t t14_generic_cell;
static volatile uint64_t t14_sink = 0x3141592653589793ULL;

__attribute__((noinline)) static T14Pair *t14_allocate_pair(uint64_t first,
                                                             uint64_t second)
{
    T14Pair *pair = malloc(sizeof(*pair));
    if (pair == NULL) {
        _exit(77);
    }
    pair->first = first;
    pair->second = second;
    return pair;
}

__attribute__((constructor)) static void t14_initialize(void)
{
    t14_instances[0] = t14_allocate_pair(0x3333333333333333ULL,
                                         0x4444444444444444ULL);
    t14_instances[1] = t14_allocate_pair(0x1111111111111111ULL,
                                         0x2222222222222222ULL);
    t14_pointer_cell = t14_instances[1];
}

__attribute__((noinline)) static uint64_t t14_read_pointer_cell(void)
{
    T14Pair *pointer = t14_pointer_cell;
    if (pointer == NULL || pointer != t14_instances[1]) return 0;
    return pointer->first ^ pointer->second;
}

__attribute__((noinline)) static uint64_t t14_read_late_pointer_cell(void)
{
    T14Pair *pointer = t14_late_pointer_cell;
    if (pointer == NULL || pointer != t14_instances[1]) return 0;
    return pointer->first + pointer->second;
}

int main(void)
{
    if (t14_null_cell != NULL) {
        t14_sink ^= t14_null_cell->first;
    }
    t14_sink ^= t14_generic_cell;
    t14_sink ^= t14_read_pointer_cell();
    t14_late_pointer_cell = t14_instances[1];
    t14_sink ^= t14_read_late_pointer_cell();
    (void)t14_sink;
    _exit(0);
}

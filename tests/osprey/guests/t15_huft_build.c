/* t15_huft_build: Stage-9 structural-ranking fixture.
 *
 * One noinline function materializes the same two adjacent uint64_t fields
 * on the stack and at one malloc site.  Equal-width field copies, matching
 * pair reads, and one stack pointer cell provide independent structure and
 * points-to evidence.  A 16-byte vector access remains a competing
 * monolithic primitive interpretation.
 */
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct HuftPair {
    uint64_t first;
    uint64_t second;
} HuftPair;

typedef unsigned long long HuftVector
    __attribute__((vector_size(16), may_alias));

HuftPair *huft_build(uint64_t seed);

__attribute__((noinline)) static uint64_t huft_read_pair(
    const HuftPair *pair)
{
    return pair->first ^ pair->second;
}

__attribute__((noinline)) HuftPair *huft_build(uint64_t seed)
{
    HuftPair stack_pair = {seed, seed ^ 0x9e3779b97f4a7c15ULL};
    HuftPair *heap_pair = malloc(sizeof(*heap_pair));
    if (heap_pair == NULL) _exit(77);

    heap_pair->first = stack_pair.first;
    heap_pair->second = stack_pair.second;

    HuftPair *volatile pointer_cell = heap_pair;
    HuftVector monolithic = *(const HuftVector *)&stack_pair;
    uint64_t result = huft_read_pair(&stack_pair);
    result ^= huft_read_pair(pointer_cell);
    *(volatile HuftVector *)heap_pair = monolithic;
    if (result == 0) _exit(78);
    _exit(0);
}

int main(void)
{
    (void)huft_build(0x1122334455667788ULL);
    _exit(0);
}

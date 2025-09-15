#pragma once

#include <memory>
#include <stdint.h>
#include <sys/mman.h>

#include <cstdlib>
#include <cstring>

#include "utils/tools.hpp"

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

namespace saqlib::memory {
template <size_t alignment, class T, bool HUGE_PAGE = false>
inline T *align_mm(size_t size) {
    size_t nbytes = utils::rd_up_to_multiple_of(size * sizeof(T), alignment);
    void *p = std::aligned_alloc(alignment, nbytes);
    if (HUGE_PAGE) {
        madvise(p, nbytes, MADV_HUGEPAGE);
    }
    std::memset(p, 0, size);
    return static_cast<T *>(p);
}

template <typename T, size_t alignment = 64>
struct AlignedAllocator {
    T *ptr = nullptr;
    size_t alignment_ = alignment;
    using value_type = T;
    T *allocate(size_t n) {
        size_t nbytes = utils::rd_up_to_multiple_of(n * sizeof(T), alignment_);
        return ptr = (T *)std::aligned_alloc(alignment_, nbytes);
    }
    void deallocate(T *p, size_t) {
        std::free(p);
        p = nullptr;
    }
    template <typename U>
    struct rebind {
        typedef AlignedAllocator<U, alignment> other;
    };
    bool operator!=(const AlignedAllocator &rhs) { return alignment_ != rhs.alignment_; }
    bool operator==(const AlignedAllocator &rhs) { return alignment_ == rhs.alignment_; }
};

template <typename T>
using UniqueArray = std::unique_ptr<T[], void (*)(void *)>;

/**
 * @brief Create a unique array with the specified size and alignment. The returned UniqueArray manages the
 *        lifetime of the allocated memory, freeing it using std::free when it goes out of scope.
 * @tparam T The type of the elements in the array.
 * @param size The number of elements in the array.
 * @param alignment The alignment requirement for the array (in bytes). 64 for AVX512.
 *                  If 0, thedefault alignment of std::malloc is used.
 */
template <typename T>
UniqueArray<T> make_unique_array(std::size_t size, std::size_t alignment = 0) {
    if (size == 0) {
        return UniqueArray<T>(nullptr, [](void *) {});
    }
    void *raw_ptr;
    if (alignment) {
        raw_ptr = std::aligned_alloc(alignment, utils::rd_up_to_multiple_of(size * sizeof(T), alignment));
    } else {
        raw_ptr = std::malloc(size * sizeof(T));
    }

    if (!raw_ptr) {
        throw std::bad_alloc();
    }
    return std::unique_ptr<T[], void (*)(void *)>(static_cast<T *>(raw_ptr), [](void *ptr) {
        std::free(ptr);
    });
}

static inline void prefetch_l1(const void *addr) {
#if defined(__SSE2__)
    _mm_prefetch(addr, _MM_HINT_T0);
#else
    __builtin_prefetch(addr, 0, 3);
#endif
}

static inline void prefetch_l2(const void *addr) {
#if defined(__SSE2__)
    _mm_prefetch((const char *)addr, _MM_HINT_T1);
#else
    __builtin_prefetch(addr, 0, 2);
#endif
}

inline void mem_prefetch_l1(const char *ptr, size_t num_lines) {
    switch (num_lines) {
    default:
        [[fallthrough]];
    case 20:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 19:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 18:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 17:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 16:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 15:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 14:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 13:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 12:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 11:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 10:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 9:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 8:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 7:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 6:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 5:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 4:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 3:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 2:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 1:
        prefetch_l1(ptr);
        ptr += 64;
        [[fallthrough]];
    case 0:
        break;
    }
}

inline void mem_prefetch_l2(const char *ptr, size_t num_lines) {
    switch (num_lines) {
    default:
        [[fallthrough]];
    case 20:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 19:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 18:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 17:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 16:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 15:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 14:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 13:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 12:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 11:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 10:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 9:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 8:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 7:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 6:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 5:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 4:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 3:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 2:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 1:
        prefetch_l2(ptr);
        ptr += 64;
        [[fallthrough]];
    case 0:
        break;
    }
}
} // namespace saqlib::memory
#pragma once

#include <bit>
#include <cstdint>
#include <immintrin.h>

namespace saqlib::utils {
inline float L2Sqr16(
    const float *__restrict__ x, const float *__restrict__ y, size_t L) {
    float result = 0;
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < L; i += 16) {
        __m512 xx = _mm512_loadu_ps(&x[i]);
        __m512 yy = _mm512_loadu_ps(&y[i]);
        __m512 t = _mm512_sub_ps(xx, yy);
        sum = _mm512_fmadd_ps(t, t, sum);
    }
    result = _mm512_reduce_add_ps(sum);
    return result;
#else
    for (size_t i = 0; i < L; ++i) {
        float tmp = x[i] - y[i];
        result += tmp * tmp;
    }
    return result;
#endif
}

inline float L2Sqr(
    const float *__restrict__ x, const float *__restrict__ y, size_t L) {
    size_t num16 = L - (L & 0b1111);
    float result = L2Sqr16(x, y, num16);
    for (size_t i = num16; i < L; ++i) {
        float tmp = x[i] - y[i];
        result += tmp * tmp;
    }
    return result;
}

/* Compute L2sqr to origin */
inline float L2Sqr(const float *__restrict__ x, size_t L) {
    float result = 0;
#if defined(__AVX512F__)
    size_t num16 = L - (L & 0b1111);
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < num16; i += 16) {
        __m512 xx = _mm512_loadu_ps(&x[i]);
        sum = _mm512_fmadd_ps(xx, xx, sum);
    }
    result = _mm512_reduce_add_ps(sum);
    for (size_t i = num16; i < L; ++i) {
        float tmp = x[i];
        result += tmp * tmp;
    }
    return result;
#else
    for (size_t i = 0; i < L; ++i) {
        float tmp = x[i];
        result += tmp * tmp;
    }
    return result;
#endif
}

inline float IP(const float *x, const float *y, size_t L) {
    float result = 0;
#if defined(__AVX512F__)
    size_t num16 = L - (L & 0b1111);
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < num16; i += 16) {
        __m512 xx = _mm512_loadu_ps(&x[i]);
        __m512 yy = _mm512_loadu_ps(&y[i]);
        sum = _mm512_fmadd_ps(xx, yy, sum);
    }
    result = _mm512_reduce_add_ps(sum);
    for (size_t i = num16; i < L; ++i) {
        result += x[i] * y[i];
    }
    return result;
#else
    for (size_t i = 0; i < L; ++i) {
        result += x[i] * y[i];
    }
    return result;
#endif
}

inline uint32_t reverse_bits(uint32_t n) {
    n = ((n >> 1) & 0x55555555) | ((n << 1) & 0xaaaaaaaa);
    n = ((n >> 2) & 0x33333333) | ((n << 2) & 0xcccccccc);
    n = ((n >> 4) & 0x0f0f0f0f) | ((n << 4) & 0xf0f0f0f0);
    n = ((n >> 8) & 0x00ff00ff) | ((n << 8) & 0xff00ff00);
    n = ((n >> 16) & 0x0000ffff) | ((n << 16) & 0xffff0000);
    return n;
}
inline uint64_t reverse_bits_u64(uint64_t n) {
    n = ((n >> 1) & 0x5555555555555555) | ((n << 1) & 0xaaaaaaaaaaaaaaaa);
    n = ((n >> 2) & 0x3333333333333333) | ((n << 2) & 0xcccccccccccccccc);
    n = ((n >> 4) & 0x0f0f0f0f0f0f0f0f) | ((n << 4) & 0xf0f0f0f0f0f0f0f0);
    n = ((n >> 8) & 0x00ff00ff00ff00ff) | ((n << 8) & 0xff00ff00ff00ff00);
    n = ((n >> 16) & 0x0000ffff0000ffff) | ((n << 16) & 0xffff0000ffff0000);
    n = ((n >> 32) & 0x00000000ffffffff) | ((n << 32) & 0xffffffff00000000);
    return n;
}

// Helper function to compute popcount for AVX512 vectors with fallback
inline __m512i avx512_popcnt_epi64(__m512i x_vec) {
#ifdef __AVX512VPOPCNTDQ__
    // Use hardware instruction if available
    return _mm512_popcnt_epi64(x_vec);
#else
    // Fallback implementation using scalar popcount
    uint64_t x_arr[8];
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(x_arr), x_vec);

    uint64_t popcnt_arr[8];
    for (int k = 0; k < 8; k++) {
        popcnt_arr[k] = __builtin_popcountll(x_arr[k]);
    }

    return _mm512_loadu_si512(reinterpret_cast<const __m512i *>(popcnt_arr));
#endif
}

inline float warmup_ip_x0_q(
    const uint64_t *data,  // pointer to data blocks (each 64 bits)
    const uint64_t *query, // pointer to query words (each 64 bits), arranged so that for
                           // each data block the corresponding b_query query words follow
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query = 0 // not used
) {
    const size_t num_blk = padded_dim / 64;
    size_t ip_scalar = 0;
    size_t ppc_scalar = 0;

    // Process blocks in chunks of 8
    const size_t vec_width = 8;
    size_t vec_end = (num_blk / vec_width) * vec_width;

    // Vector accumulators (each holds 8 64-bit lanes)
    __m512i ip_vec = _mm512_setzero_si512();  // will accumulate weighted popcount intersections per block
    __m512i ppc_vec = _mm512_setzero_si512(); // will accumulate popcounts of data blocks

    // Loop over blocks in batches of 8
    for (size_t i = 0; i < vec_end; i += vec_width) {
        // Load eight 64-bit data blocks into x_vec.
        __m512i x_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(data + i));

        // Compute popcount for each 64-bit block in x_vec using the AVX512 VPOPCNTDQ
        // instruction. (Ensure you compile with the proper flags for VPOPCNTDQ.)
        __m512i popcnt_x_vec;
        {
#ifdef __AVX512VPOPCNTDQ__
            // Use hardware instruction if available
            popcnt_x_vec = _mm512_popcnt_epi64(x_vec);
#else
            // Fallback implementation using scalar popcount
            uint64_t popcnt_arr[8];
            for (int k = 0; k < 8; k++) {
                popcnt_arr[k] = std::popcount(data[i + k]);
            }
            popcnt_x_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(popcnt_arr));
#endif
        }
        ppc_vec = _mm512_add_epi64(ppc_vec, popcnt_x_vec);

        // For accumulating the weighted popcounts per block.
        __m512i block_ip = _mm512_setzero_si512();

        // Process each query component (b_query is a compile-time constant, and is small).
        for (uint32_t j = 0; j < b_query; j++) {
            // We need to gather from query array the j-th query for each of the eight
            // blocks. For block (i + k) the index is: ( (i + k) * b_query + j ). We
            // construct an index vector of eight 64-bit indices.
            uint64_t indices[vec_width];
            for (size_t k = 0; k < vec_width; k++) {
                indices[k] = ((i + k) * b_query + j);
            }
            // Load indices from memory.
            __m512i index_vec = _mm512_loadu_si512(indices);
            // Gather 8 query words with a scale of 8 (since query is an array of 64-bit
            // integers).
            __m512i q_vec = _mm512_i64gather_epi64(index_vec, query, 8);

            // Compute bitwise AND of data blocks and corresponding query words.
            __m512i and_vec = _mm512_and_si512(x_vec, q_vec);
            // Compute popcount on each lane.
            __m512i popcnt_and = avx512_popcnt_epi64(and_vec);

            // Multiply by the weighting factor (1 << j) for this query position.
            const uint64_t shift = 1ULL << j;
            __m512i shift_vec = _mm512_set1_epi64(shift);
            __m512i weighted = _mm512_mullo_epi64(popcnt_and, shift_vec);

            // Accumulate weighted popcounts for these blocks.
            block_ip = _mm512_add_epi64(block_ip, weighted);
        }
        // Add the block's query-weighted popcount to the overall ip vector.
        ip_vec = _mm512_add_epi64(ip_vec, block_ip);
    }

    // Horizontally reduce the vector accumulators.
    uint64_t ip_arr[vec_width];
    uint64_t ppc_arr[vec_width];
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(ip_arr), ip_vec);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(ppc_arr), ppc_vec);

    for (size_t k = 0; k < vec_width; k++) {
        ip_scalar += ip_arr[k];
        ppc_scalar += ppc_arr[k];
    }

    // Process remaining blocks that did not fit in the vectorized loop.
    for (size_t i = vec_end; i < num_blk; i++) {
        const uint64_t x = data[i];
        ppc_scalar += std::popcount(x);
        for (uint32_t j = 0; j < b_query; j++) {
            ip_scalar += std::popcount(x & query[i * b_query + j]) << j;
        }
    }

    return (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar));
}

inline float mask_ip_x0_q(const float *query, const uint64_t *data, size_t padded_dim) {
    const size_t num_blk = padded_dim / 64;
    const uint64_t *it_data = data;
    const float *it_query = query;

    //    __m512 sum0 = _mm512_setzero_ps();
    //    __m512 sum1 = _mm512_setzero_ps();
    //    __m512 sum2 = _mm512_setzero_ps();
    //    __m512 sum3 = _mm512_setzero_ps();

    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);

        __mmask16 mask0 = static_cast<__mmask16>(bits);
        __mmask16 mask1 = static_cast<__mmask16>(bits >> 16);
        __mmask16 mask2 = static_cast<__mmask16>(bits >> 32);
        __mmask16 mask3 = static_cast<__mmask16>(bits >> 48);

        __m512 masked0 = _mm512_maskz_loadu_ps(mask0, it_query);
        __m512 masked1 = _mm512_maskz_loadu_ps(mask1, it_query + 16);
        __m512 masked2 = _mm512_maskz_loadu_ps(mask2, it_query + 32);
        __m512 masked3 = _mm512_maskz_loadu_ps(mask3, it_query + 48);

        sum = _mm512_add_ps(sum, masked0);
        sum = _mm512_add_ps(sum, masked1);
        sum = _mm512_add_ps(sum, masked2);
        sum = _mm512_add_ps(sum, masked3);

        //         _mm_prefetch(reinterpret_cast<const char*>(it_query + 128), _MM_HINT_T1);

        ++it_data;
        it_query += 64;
    }

    //    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    return _mm512_reduce_add_ps(sum);
}
inline void new_transpose_bin(
    const uint16_t *q, uint64_t *tq, size_t padded_dim, size_t b_query) {
    // 512 / 16 = 32
    for (size_t i = 0; i < padded_dim; i += 64) {
        __m512i vec_00_to_31 = _mm512_loadu_si512(q);
        __m512i vec_32_to_63 = _mm512_loadu_si512(q + 32);

        // the first (16 - b_query) bits are empty
        vec_00_to_31 = _mm512_slli_epi32(vec_00_to_31, (16 - b_query));
        vec_32_to_63 = _mm512_slli_epi32(vec_32_to_63, (16 - b_query));

        for (size_t j = 0; j < b_query; ++j) {
            uint32_t v0 = _mm512_movepi16_mask(vec_00_to_31); // get most significant bit
            uint32_t v1 = _mm512_movepi16_mask(vec_32_to_63); // get most significant bit
            // [TODO: remove all reverse_bits]
            v0 = reverse_bits(v0);
            v1 = reverse_bits(v1);
            uint64_t v = (static_cast<uint64_t>(v0) << 32) + v1;

            tq[b_query - j - 1] = v;

            vec_00_to_31 = _mm512_slli_epi16(vec_00_to_31, 1);
            vec_32_to_63 = _mm512_slli_epi16(vec_32_to_63, 1);
        }
        tq += b_query;
        q += 64;
    }
}

} // namespace saqlib::utils
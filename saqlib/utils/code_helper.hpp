#pragma once

#include <bits/stdc++.h>
#include <cstddef>
#include <immintrin.h>

#include <cstdint>
#include <memory>

#include "utils/memory.hpp"
#include "utils/space.hpp"

namespace saqlib::utils {

template <size_t kBits>
class CodeHelper {
    template <typename T>
    static void froce_compact(uint8_t *o_compact, T *o_raw, size_t num_dim) {
        for (size_t i = 0; i < num_dim * kBits / 8; i++) {
            o_compact[i] = 0;
        }
        size_t shift = 0;
        for (size_t d = 0; d < num_dim; d += 1) {
            auto t = o_raw[d];
            for (size_t i = 0; i < kBits; i++, t >>= 1) {
                o_compact[i] |= (t & 1) << shift;
            }
            ++shift;
            if (shift == 8) {
                shift = 0;
                o_compact += kBits;
            }
        }
        assert(shift == 0);
    }

    template <typename T>
    static void froce_decompact(const uint8_t *__restrict__ y, T *out, size_t D) {
        if (kBits == 0)
            return;
        uint8_t shift_v = 1;
        size_t y_p = 0;
        for (size_t d = 0; d < D; d++) {
            out[d] = 0;
            for (size_t i = 0; i < kBits; i++) {
                out[d] |= ((y[y_p + i] & shift_v) != 0) << i;
            }
            shift_v <<= 1;
            if (shift_v == 0) {
                shift_v = 1;
                y_p += kBits;
            }
        }
        assert(y_p == D * kBits / 8);
    }

  public:
    static void compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
        froce_compact(o_compact, o_raw, num_dim);
    }

    static void compacted_code16(uint8_t *o_compact, const uint16_t *o_raw16, size_t num_dim) {
        if constexpr (kBits == 0) {
            return;
        }
        auto o_raw8 = std::make_unique<uint8_t[]>(num_dim);
        if (kBits > 8) {
            for (size_t i = 0; i < num_dim; i++) {
                o_compact[i] = o_raw16[i] & 0xFF;
            }
            o_compact += num_dim;
            for (size_t i = 0; i < num_dim; i++) {
                o_raw8[i] = o_raw16[i] >> 8;
            }
            CodeHelper<kBits - 8>::compacted_code8(o_compact, o_raw8.get(), num_dim);
        } else {
            std::copy(o_raw16, o_raw16 + num_dim, o_raw8.get());
            compacted_code8(o_compact, o_raw8.get(), num_dim);
        }
    }

    static float compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ y, size_t D) {
        if constexpr (kBits == 0)
            return 0;
        if constexpr (kBits > 8) {
            auto ip = CodeHelper<8>::compute_ip(query, y, D);
            return ip + 256 * CodeHelper<kBits - 8>::compute_ip(query, y + D, D);
        }
        auto rec = memory::make_unique_array<uint8_t>(D, 64);
        froce_decompact(y, rec.get(), D);
        return CodeHelper<8>::compute_ip(query, rec.get(), D);
    }
};

template <>
inline void CodeHelper<1>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    for (size_t i = 0; i < num_dim; i += 8) {
        o_compact[i / 8] = 0;
        for (size_t j = 0; j < 8; j++) {
            o_compact[i / 8] |= ((o_raw[i + j] & 1) << j);
        }
    }
}

template <>
inline float CodeHelper<1>::compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ mask, size_t len) {
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps(); // Initialize the accumulator to 0

    // Process 16 float elements per loop
    for (size_t i = 0; i < len; i += 16) {
        // Calculate the mask position corresponding to the current block
        const auto m = mask + (i / 8);
        __mmask16 k = _cvtu32_mask16(m[0] + ((uint32_t)m[1] << 8));

        // Load data based on the mask (unselected positions are set to 0)
        __m512 values = _mm512_maskz_load_ps(k, query + i);

        // Accumulate to the accumulator
        acc = _mm512_add_ps(acc, values);
    }

    // Sum all elements in the accumulator
    return _mm512_reduce_add_ps(acc);
#else
    float ans = 0;
    for (size_t i = 0; i < len; i += 8) {
        auto m = mask[(i / 8)];
        for (int j = 0; j < 8; ++j) {
            if (m & (1 << j)) {
                ans += x[i + j];
            }
        }
    }
    return ans;
#endif
}

template <>
inline void CodeHelper<2>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    // Create a mask to isolate the two least significant bits of each byte
    __m128i mask = _mm_set1_epi8(0b00000011);

    // Process the data in chunks of 64 bytes
    for (size_t d = 0; d < num_dim; d += 64) {
        // Load 64 bytes of raw data into four 128-bit vectors
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw));
        __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 16));
        __m128i vec_32_to_47 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 32));
        __m128i vec_48_to_63 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 48));

        // Apply the mask to extract the two least significant bits from each vector
        vec_00_to_15 = _mm_and_si128(vec_00_to_15, mask);
        vec_16_to_31 = _mm_slli_epi16(_mm_and_si128(vec_16_to_31, mask), 2); // Shift left by 2 bits
        vec_32_to_47 = _mm_slli_epi16(_mm_and_si128(vec_32_to_47, mask), 4); // Shift left by 4 bits
        vec_48_to_63 = _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask), 6); // Shift left by 6 bits

        // Combine the processed vectors into a single compact representation
        __m128i compact = _mm_or_si128(
            _mm_or_si128(vec_00_to_15, vec_16_to_31),
            _mm_or_si128(vec_32_to_47, vec_48_to_63));

        // Store the compacted data into the output buffer
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact), compact);

        // Move to the next chunk of raw data and output buffer
        o_raw += 64;
        o_compact += 16;
    }
}

template <>
inline float CodeHelper<2>::compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ y, size_t D) {
    __m512 sum = _mm512_setzero_ps();
    uint8_t *o_compact = const_cast<uint8_t *>(y);
    float result = 0;

    __m128i mask = _mm_set1_epi8(0b00000011);

    for (size_t i = 0; i < D; i += 64) {
        __m128i cpt = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact));

        __m128i vec_00_to_15 = _mm_and_si128(cpt, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(cpt, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(cpt, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(cpt, 6), mask);

        __m512 xx, yy;

        xx = _mm512_loadu_ps(&query[i]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 16]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 32]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 48]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        o_compact += 16;
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

template <>
inline void CodeHelper<3>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    // Create a mask to isolate the two least significant bits of each byte
    __m128i mask = _mm_set1_epi8(0b11);
    // __m128i top_mask = _mm_set1_epi8(0b100);

    for (size_t d = 0; d < num_dim; d += 64) {
        // Load 64 bytes of raw data into four 128-bit vectors
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw));
        __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 16));
        __m128i vec_32_to_47 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 32));
        __m128i vec_48_to_63 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 48));

        // Apply the mask to extract the two least significant bits from each vector
        vec_00_to_15 = _mm_and_si128(vec_00_to_15, mask);
        vec_16_to_31 = _mm_slli_epi16(_mm_and_si128(vec_16_to_31, mask), 2);
        vec_32_to_47 = _mm_slli_epi16(_mm_and_si128(vec_32_to_47, mask), 4);
        vec_48_to_63 = _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask), 6);

        // Combine the processed vectors into a single compact representation
        __m128i compact = _mm_or_si128(
            _mm_or_si128(vec_00_to_15, vec_16_to_31),
            _mm_or_si128(vec_32_to_47, vec_48_to_63));

        // Store the compacted data into the output buffer
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact), compact);
        o_compact += 16;

        // Initialize top_bit to store the top bits of the raw data
        int64_t top_bit = 0;
        int64_t top_mask = 0x0101010101010101;
        // Extract the top bits from the raw data
        for (size_t i = 0; i < 64; i += 8) {
            int64_t cur_codes = *reinterpret_cast<const int64_t *>(o_raw + i);
            top_bit |= ((cur_codes >> 2) & top_mask) << (i / 8);
        }
        // Copy the top bits to the output buffer
        std::memcpy(o_compact, &top_bit, sizeof(int64_t));

        o_raw += 64;
        o_compact += 8;
    }
}

template <>
inline float CodeHelper<3>::compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ y, size_t D) {
    __m512 sum = _mm512_setzero_ps();
    uint8_t *o_compact = const_cast<uint8_t *>(y);
    float result = 0;

    __m128i mask = _mm_set1_epi8(0b11);
    __m128i top_mask = _mm_set1_epi8(0b100);

    for (size_t i = 0; i < D; i += 64) {
        __m128i cpt = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact));
        o_compact += 16;

        int64_t top_bit = *reinterpret_cast<int64_t *>(o_compact);
        o_compact += 8;

        __m128i vec_00_to_15 = _mm_and_si128(cpt, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(cpt, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(cpt, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(cpt, 6), mask);

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit >> 0), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 3, top_bit >> 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 5, top_bit >> 4), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m512 xx, yy;

        xx = _mm512_loadu_ps(&query[i]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 16]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 32]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 48]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

template <>
inline void CodeHelper<4>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    for (size_t j = 0; j < num_dim; j += 32) {
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw));
        __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 16));
        vec_16_to_31 = _mm_slli_epi16(vec_16_to_31, 4);

        __m128i compact = _mm_or_si128(vec_00_to_15, vec_16_to_31);

        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact), compact);

        o_raw += 32;
        o_compact += 16;
    }
}

template <>
inline float CodeHelper<4>::compute_ip(const float *__restrict__ x, const uint8_t *__restrict__ y, size_t D) {
    __m128i mask = _mm_set1_epi8(0b1111);
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < D; i += 32) {
        __m128i a8 = _mm_loadu_epi32(&y[i / 2]);
        __m128i b8 = a8;
        __m512 x1 = _mm512_load_ps(&x[i]);
        __m512 x2 = _mm512_load_ps(&x[i + 16]);

        // get lower(0 to 15) and upper(16 to 31) 4 bits
        a8 = _mm_and_si128(a8, mask);
        b8 = _mm_and_si128(_mm_srli_epi16(b8, 4), mask);

        __m512 af = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a8));
        sum = _mm512_fmadd_ps(af, x1, sum);
        __m512 bf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b8));
        sum = _mm512_fmadd_ps(bf, x2, sum);
    }
    return _mm512_reduce_add_ps(sum);
}

template <>
inline void CodeHelper<5>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    CodeHelper<1>::compacted_code8(o_compact + (num_dim * 4 / 8), o_raw, num_dim);
    auto o4 = memory::make_unique_array<uint8_t>(num_dim, 16);
    for (size_t i = 0; i < num_dim; i++) {
        o4[i] = o_raw[i] >> 1;
    }
    CodeHelper<4>::compacted_code8(o_compact, o4.get(), num_dim);
}

template <>
inline float CodeHelper<5>::compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ y, size_t D) {
    return 2 * CodeHelper<4>::compute_ip(query, y, D) + CodeHelper<1>::compute_ip(query, y + (D * 4 / 8), D);
}

template <>
inline void CodeHelper<6>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    __m128i mask2 = _mm_set1_epi8(0b11000000);
    __m128i mask4 = _mm_set1_epi8(0b00001111);
    for (size_t d = 0; d < num_dim; d += 64) {
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw));
        __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 16));
        __m128i vec_32_to_47 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 32));
        __m128i vec_48_to_63 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 48));

        __m128i compact = _mm_or_si128(
            vec_00_to_15, _mm_and_si128(_mm_slli_epi16(vec_32_to_47, 2), mask2));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact + 0), compact);

        compact = _mm_or_si128(
            vec_16_to_31, _mm_and_si128(_mm_slli_epi16(vec_48_to_63, 2), mask2));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact + 16), compact);

        compact = _mm_or_si128(
            _mm_and_si128(vec_32_to_47, mask4),
            _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask4), 4));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact + 32), compact);

        o_raw += 64;
        o_compact += 48;
    }
}

template <>
inline float CodeHelper<6>::compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ y, size_t D) {
    __m512 sum = _mm512_setzero_ps();
    uint8_t *o_compact = const_cast<uint8_t *>(y);
    float result = 0;

    __m128i mask6 = _mm_set1_epi8(0b00111111);
    __m128i mask2 = _mm_set1_epi8(0b00110000);
    __m128i mask4 = _mm_set1_epi8(0b00001111);

    for (size_t i = 0; i < D; i += 64) {
        __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact + 0));
        __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact + 16));
        __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact + 32));

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(cpt1, 2), mask2), _mm_and_si128(cpt3, mask4));
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(cpt2, 2), mask2),
            _mm_and_si128(_mm_srli_epi16(cpt3, 4), mask4));

        __m512 xx, yy;

        xx = _mm512_loadu_ps(&query[i]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 16]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 32]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 48]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        o_compact += 48;
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

template <>
inline void CodeHelper<7>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    __m128i mask2 = _mm_set1_epi8(0b11000000);
    __m128i mask4 = _mm_set1_epi8(0b00001111);
    __m128i mask6 = _mm_set1_epi8(0b00111111);
    for (size_t d = 0; d < num_dim; d += 64) {
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw));
        __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 16));
        __m128i vec_32_to_47 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 32));
        __m128i vec_48_to_63 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 48));

        __m128i compact = _mm_or_si128(
            _mm_and_si128(vec_00_to_15, mask6),
            _mm_and_si128(_mm_slli_epi16(vec_32_to_47, 2), mask2));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact + 0), compact);

        compact = _mm_or_si128(
            _mm_and_si128(vec_16_to_31, mask6),
            _mm_and_si128(_mm_slli_epi16(vec_48_to_63, 2), mask2));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact + 16), compact);

        compact = _mm_or_si128(
            _mm_and_si128(vec_32_to_47, mask4),
            _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask4), 4));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact + 32), compact);
        o_compact += 48;

        int64_t top_bit = 0;
        int64_t top_mask = 0x0101010101010101;
        for (size_t i = 0; i < 64; i += 8) {
            int64_t cur_codes = *reinterpret_cast<const int64_t *>(o_raw + i);
            top_bit |= ((cur_codes >> 6) & top_mask) << (i / 8);
        }
        std::memcpy(o_compact, &top_bit, sizeof(int64_t));

        o_compact += 8;
        o_raw += 64;
    }
}

template <>
inline float CodeHelper<7>::compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ y, size_t D) {
    __m512 sum = _mm512_setzero_ps();
    uint8_t *o_compact = const_cast<uint8_t *>(y);
    float result = 0;

    __m128i mask6 = _mm_set1_epi8(0b00111111);
    __m128i mask2 = _mm_set1_epi8(0b00110000);
    __m128i mask4 = _mm_set1_epi8(0b00001111);
    __m128i top_mask = _mm_set1_epi8(0b1000000);

    for (size_t i = 0; i < D; i += 64) {
        __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact + 0));
        __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact + 16));
        __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact + 32));

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(cpt1, 2), mask2), _mm_and_si128(cpt3, mask4));
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(cpt2, 2), mask2),
            _mm_and_si128(_mm_srli_epi16(cpt3, 4), mask4));
        o_compact += 48;

        int64_t top_bit = *reinterpret_cast<int64_t *>(o_compact);
        o_compact += 8;

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 5, top_bit << 6), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 3, top_bit << 4), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit << 0), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m512 xx, yy;

        xx = _mm512_loadu_ps(&query[i]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 16]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 32]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 48]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

template <>
inline void CodeHelper<8>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    std::memcpy(o_compact, o_raw, sizeof(uint8_t) * num_dim);
}

template <>
inline float CodeHelper<8>::compute_ip(const float *__restrict__ x, const uint8_t *__restrict__ y, size_t D) {
    float result = 0;
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < D; i += 16) {
        __m512 xx = _mm512_load_ps(&x[i]);
        __m512 yy =
            _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i *)&y[i])));
        sum = _mm512_fmadd_ps(xx, yy, sum);
    }
    result = _mm512_reduce_add_ps(sum);
#else
    for (size_t i = 0; i < L; i += 4) {
        result += x[i] * static_cast<float>(y[i]);
        result += x[i + 1] * static_cast<float>(y[i + 1]);
        result += x[i + 2] * static_cast<float>(y[i + 2]);
        result += x[i + 3] * static_cast<float>(y[i + 3]);
    }
#endif
    return result;
}

// template <size_t bits>
// inline void CodeHelper<bits>::compacted_code16(uint8_t *o_compact, const uint16_t *o_raw16, size_t num_dim)

// template <size_t bits>
// inline float CodeHelper<bits>::compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ y, size_t D)

inline auto get_IP_FUNC(int bits) -> float (*)(const float *__restrict__, const uint8_t *__restrict__, size_t) {
    switch (bits) {
    case 0:
        return CodeHelper<0>::compute_ip;
    case 1:
        return CodeHelper<1>::compute_ip;
    case 2:
        return CodeHelper<2>::compute_ip;
    case 3:
        return CodeHelper<3>::compute_ip;
    case 4:
        return CodeHelper<4>::compute_ip;
    case 5:
        return CodeHelper<5>::compute_ip;
    case 6:
        return CodeHelper<6>::compute_ip;
    case 7:
        return CodeHelper<7>::compute_ip;
    case 8:
        return CodeHelper<8>::compute_ip;
    case 9:
        return CodeHelper<9>::compute_ip;
    case 10:
        return CodeHelper<10>::compute_ip;
    case 11:
        return CodeHelper<11>::compute_ip;
    case 12:
        return CodeHelper<12>::compute_ip;
    case 13:
        return CodeHelper<13>::compute_ip;
    case 14:
        return CodeHelper<14>::compute_ip;
    case 15:
        return CodeHelper<15>::compute_ip;
    case 16:
        return CodeHelper<16>::compute_ip;
    default:
        std::cerr << "Error: Unsupported bits: " << bits << std::endl;
        assert(false);
    }
    return nullptr;
}

inline auto get_compacted_code16_func(int bits) -> void (*)(uint8_t *o_compact, const uint16_t *o_raw, size_t num_dim) {
    switch (bits) {
    case 0:
        return CodeHelper<0>::compacted_code16;
    case 1:
        return CodeHelper<1>::compacted_code16;
    case 2:
        return CodeHelper<2>::compacted_code16;
    case 3:
        return CodeHelper<3>::compacted_code16;
    case 4:
        return CodeHelper<4>::compacted_code16;
    case 5:
        return CodeHelper<5>::compacted_code16;
    case 6:
        return CodeHelper<6>::compacted_code16;
    case 7:
        return CodeHelper<7>::compacted_code16;
    case 8:
        return CodeHelper<8>::compacted_code16;
    case 9:
        return CodeHelper<9>::compacted_code16;
    case 10:
        return CodeHelper<10>::compacted_code16;
    case 11:
        return CodeHelper<11>::compacted_code16;
    case 12:
        return CodeHelper<12>::compacted_code16;
    case 13:
        return CodeHelper<13>::compacted_code16;
    case 14:
        return CodeHelper<14>::compacted_code16;
    case 15:
        return CodeHelper<15>::compacted_code16;
    case 16:
        return CodeHelper<16>::compacted_code16;
    default:
        assert(false);
    }
    return nullptr;
}

inline auto get_compacted_code8_func(int bits) -> void (*)(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    switch (bits) {
    case 0:
        return CodeHelper<0>::compacted_code8;
    case 1:
        return CodeHelper<1>::compacted_code8;
    case 2:
        return CodeHelper<2>::compacted_code8;
    case 3:
        return CodeHelper<3>::compacted_code8;
    case 4:
        return CodeHelper<4>::compacted_code8;
    case 5:
        return CodeHelper<5>::compacted_code8;
    case 6:
        return CodeHelper<6>::compacted_code8;
    case 7:
        return CodeHelper<7>::compacted_code8;
    case 8:
        return CodeHelper<8>::compacted_code8;
    case 9:
        return CodeHelper<9>::compacted_code8;
    case 10:
        return CodeHelper<10>::compacted_code8;
    case 11:
        return CodeHelper<11>::compacted_code8;
    case 12:
        return CodeHelper<12>::compacted_code8;
    case 13:
        return CodeHelper<13>::compacted_code8;
    case 14:
        return CodeHelper<14>::compacted_code8;
    case 15:
        return CodeHelper<15>::compacted_code8;
    case 16:
        return CodeHelper<16>::compacted_code8;
    default:
        assert(false);
    }
    return nullptr;
}
} // namespace saqlib::utils

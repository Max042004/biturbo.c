/* test_tmac.c — verify T-MAC TL2 encoding and GEMV correctness */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "biturbo.h"

/* Re-declare internal functions we need (defined static in biturbo.c) */
/* We'll compile biturbo.c with -DTEST_TMAC to expose them.
 * Instead, let's just include biturbo.c directly for testing. */

/* Pull in the implementation */
#include "biturbo.c"

/* ================================================================
 * Test 1: Encoding table correctness
 * For all 27 ternary triples, verify that the T-MAC encoding
 * produces the correct dot product via LUT lookup.
 * ================================================================ */
static int test_encoding_table(void) {
    printf("Test 1: Encoding table correctness... ");
    int errors = 0;

    /* Three-weight LUT coefficient table (what each nibble index means) */
    static const int8_t coeff[16][3] = {
        { 0, 0, 0}, { 0, 0, 1}, { 0, 1, 0}, { 1, 0, 0},
        { 0, 1, 1}, { 1, 0, 1}, { 1, 1, 0}, { 0, 1,-1},
        { 1, 0,-1}, { 1,-1, 0}, { 1, 1, 1}, { 1, 1,-1},
        { 1,-1, 1}, { 1,-1,-1}, { 0, 0, 0}, { 0, 0, 0}
    };

    /* Test with several activation triples */
    int16_t test_acts[][3] = {
        {1, 2, 3}, {-5, 10, -7}, {127, -127, 0}, {42, 42, 42},
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}
    };
    int n_tests = sizeof(test_acts) / sizeof(test_acts[0]);

    for (int w0 = -1; w0 <= 1; w0++) {
        for (int w1 = -1; w1 <= 1; w1++) {
            for (int w2 = -1; w2 <= 1; w2++) {
                uint8_t enc = TMAC3_ENC[w0+1][w1+1][w2+1];
                uint8_t nib  = enc >> 1;
                uint8_t sign = enc & 1;

                if (nib >= 14 && !(w0 == 0 && w1 == 0 && w2 == 0)) {
                    printf("FAIL: (%d,%d,%d) → nibble=%d (out of range)\n",
                           w0, w1, w2, nib);
                    errors++;
                    continue;
                }

                for (int t = 0; t < n_tests; t++) {
                    int16_t a0 = test_acts[t][0];
                    int16_t a1 = test_acts[t][1];
                    int16_t a2 = test_acts[t][2];

                    /* Expected: direct dot product */
                    int expected = w0 * a0 + w1 * a1 + w2 * a2;

                    /* Via LUT: coeff lookup, then sign correction */
                    int lut_val = coeff[nib][0] * a0 +
                                  coeff[nib][1] * a1 +
                                  coeff[nib][2] * a2;
                    int actual = sign ? -lut_val : lut_val;

                    if (actual != expected) {
                        printf("FAIL: w=(%d,%d,%d) a=(%d,%d,%d) "
                               "nib=%d sign=%d → %d, expected %d\n",
                               w0, w1, w2, a0, a1, a2,
                               nib, sign, actual, expected);
                        errors++;
                    }
                }
            }
        }
    }

    /* Two-weight encoding */
    static const int8_t coeff2[9][2] = {
        { 0, 0}, { 0, 1}, { 0,-1},
        { 1, 0}, { 1, 1}, { 1,-1},
        {-1, 0}, {-1, 1}, {-1,-1}
    };

    for (int w0 = -1; w0 <= 1; w0++) {
        for (int w1 = -1; w1 <= 1; w1++) {
            uint8_t nib = TMAC2_ENC[w0+1][w1+1];
            if (nib >= 9) {
                printf("FAIL: two-weight (%d,%d) → nibble=%d\n", w0, w1, nib);
                errors++;
                continue;
            }
            for (int t = 0; t < n_tests; t++) {
                int16_t a0 = test_acts[t][0], a1 = test_acts[t][1];
                int expected = w0 * a0 + w1 * a1;
                int actual = coeff2[nib][0] * a0 + coeff2[nib][1] * a1;
                if (actual != expected) {
                    printf("FAIL: two w=(%d,%d) a=(%d,%d) nib=%d → %d, expected %d\n",
                           w0, w1, a0, a1, nib, actual, expected);
                    errors++;
                }
            }
        }
    }

    if (errors == 0) printf("PASS (27*%d three-weight + 9*%d two-weight checks)\n",
                            n_tests, n_tests);
    else printf("%d errors\n", errors);
    return errors;
}

/* ================================================================
 * Test 2: Weight repack round-trip
 * Create a synthetic I2_S weight, repack to T-MAC, verify that
 * decoding the T-MAC nibble+sign reproduces the original ternary.
 * ================================================================ */
static int test_repack_roundtrip(void) {
    printf("Test 2: Weight repack round-trip... ");
    int errors = 0;

    /* Small synthetic weight: 4 rows, 10 cols (3*3 + 1 remainder) */
    int rows = 4, cols = 10;
    int n_blocks = (cols + 127) / 128;
    int bytes_per_row = n_blocks * 32;  /* 32 bytes for 1 block */

    /* Allocate and fill with known ternary pattern */
    uint8_t* packed = (uint8_t*)calloc((size_t)rows * bytes_per_row + 4, 1);

    /* Ternary values for each row (10 cols) */
    int8_t vals[4][10] = {
        { 1, 0,-1, 1, 1, 0, -1,-1, 1, 0},
        { 0, 0, 0, 1,-1, 1,  0, 1,-1, 1},
        {-1,-1,-1, 0, 0, 0,  1, 1, 1,-1},
        { 1, 1, 1, 1, 1, 1,  1, 1, 1, 1}
    };

    /* Pack into I2_S format: 2-bit codes, group-interleaved blocks of 128 */
    /* Code: -1→0, 0→1, +1→2 */
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int block = c / 128;
            int group = (c % 128) / 32;
            int pos   = c % 32;
            uint8_t code;
            if (vals[r][c] == -1) code = 0;
            else if (vals[r][c] == 0) code = 1;
            else code = 2;  /* +1 */
            int shift = 6 - 2 * group;
            packed[r * bytes_per_row + block * 32 + pos] |= (code << shift);
        }
    }

    /* Set scale */
    float scale = 0.5f;
    memcpy(packed + (size_t)rows * bytes_per_row, &scale, 4);

    /* Create I2_S weight */
    bt_i2s_weight_t w = {0};
    w.data = packed;
    w.scale = scale;
    w.rows = rows;
    w.cols = cols;
    w.tmac = NULL;

    /* Verify i2s_decode works */
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int8_t decoded = i2s_decode(&w, r, c);
            if (decoded != vals[r][c]) {
                printf("FAIL: i2s_decode(%d,%d) = %d, expected %d\n",
                       r, c, decoded, vals[r][c]);
                errors++;
            }
        }
    }

    /* Repack to T-MAC */
    tmac_repack(&w);
    bt_tmac_weight_t* tw = w.tmac;

    if (!tw) {
        printf("FAIL: tmac_repack returned NULL\n");
        free(packed);
        return 1;
    }

    /* Verify: decode T-MAC nibble+sign back to ternary and compare */
    /* Three-weight LUT coefficients */
    static const int8_t coeff[16][3] = {
        { 0, 0, 0}, { 0, 0, 1}, { 0, 1, 0}, { 1, 0, 0},
        { 0, 1, 1}, { 1, 0, 1}, { 1, 1, 0}, { 0, 1,-1},
        { 1, 0,-1}, { 1,-1, 0}, { 1, 1, 1}, { 1, 1,-1},
        { 1,-1, 1}, { 1,-1,-1}, { 0, 0, 0}, { 0, 0, 0}
    };

    for (int r = 0; r < rows; r++) {
        for (int g = 0; g < tw->n3; g++) {
            /* Extract nibble */
            uint8_t pb = tw->three_nib[(size_t)r * tw->nib3_stride + g/2];
            int nib = (g & 1) ? (pb & 0x0F) : (pb >> 4);

            /* Extract sign */
            int sign = (tw->three_sign[(size_t)r * tw->sign_stride + g/8] >> (g&7)) & 1;

            /* Reconstruct ternary triple */
            int8_t c0 = sign ? -coeff[nib][0] : coeff[nib][0];
            int8_t c1 = sign ? -coeff[nib][1] : coeff[nib][1];
            int8_t c2 = sign ? -coeff[nib][2] : coeff[nib][2];

            if (c0 != vals[r][g*3+0] || c1 != vals[r][g*3+1] || c2 != vals[r][g*3+2]) {
                printf("FAIL: row=%d group=%d nib=%d sign=%d → (%d,%d,%d), "
                       "expected (%d,%d,%d)\n",
                       r, g, nib, sign, c0, c1, c2,
                       vals[r][g*3+0], vals[r][g*3+1], vals[r][g*3+2]);
                errors++;
            }
        }
    }

    tmac_free(&w);
    free(packed);

    if (errors == 0) printf("PASS\n");
    else printf("%d errors\n", errors);
    return errors;
}

/* ================================================================
 * Test 3: GEMV bit-exactness
 * Run both i2s_gemv (scalar) and tmac_gemv on the same inputs,
 * verify identical float outputs.
 * ================================================================ */
static int test_gemv_bitexact(void) {
    printf("Test 3: GEMV bit-exactness... ");
    int errors = 0;

    /* Create a larger synthetic weight: 8 rows, 15 cols (5 three-groups + 0 two) */
    /* Actually 15/3=5, remainder 0. Try 16 cols: 5 three-groups + 1 two-group */
    int rows = 8, cols = 16;
    int n_blocks = (cols + 127) / 128;
    int bytes_per_row = n_blocks * 32;

    uint8_t* packed = (uint8_t*)calloc((size_t)rows * bytes_per_row + 4, 1);

    /* Random-ish ternary weights */
    int8_t wvals[8][16] = {
        { 1, 0,-1, 1, 1, 0,-1,-1, 1, 0, 1,-1, 0, 1,-1, 1},
        { 0, 0, 0, 1,-1, 1, 0, 1,-1, 1, 0, 0, 1, 0, 0,-1},
        {-1,-1,-1, 0, 0, 0, 1, 1, 1,-1, 1, 0,-1, 0, 1, 0},
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1},
        { 0, 1, 0,-1, 0, 1, 0,-1, 0, 1, 0,-1, 0, 1, 0,-1},
        { 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
    };

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int block = c / 128;
            int group = (c % 128) / 32;
            int pos   = c % 32;
            uint8_t code;
            if (wvals[r][c] == -1) code = 0;
            else if (wvals[r][c] == 0) code = 1;
            else code = 2;
            packed[r * bytes_per_row + block * 32 + pos] |= (code << (6 - 2*group));
        }
    }

    float scale = 0.123f;
    memcpy(packed + (size_t)rows * bytes_per_row, &scale, 4);

    bt_i2s_weight_t w = {0};
    w.data = packed;
    w.scale = scale;
    w.rows = rows;
    w.cols = cols;
    w.tmac = NULL;

    /* INT8 activations */
    int8_t x_q[16] = {10, -20, 30, -40, 50, -60, 70, -80,
                       90, -100, 110, -120, 127, -127, 42, -7};
    float inv_scale = 0.5f;

    /* Scalar i2s_gemv */
    float out_scalar[8];
    i2s_gemv(out_scalar, &w, x_q, inv_scale);

    /* T-MAC path */
    tmac_repack(&w);
    bt_tmac_weight_t* tw = w.tmac;

    int16_t lut_buf[(16/3 + 1) * 16];
    memset(lut_buf, 0, sizeof(lut_buf));
    tmac_build_three_lut(lut_buf, x_q, tw->n3);
    int16_t* two_lut = lut_buf + tw->n3 * 16;
    tmac_build_two_lut(two_lut, x_q, tw->n3 * 3, cols, tw->n2);

    float out_tmac[8];
    tmac_gemv(out_tmac, tw, inv_scale, lut_buf, two_lut);

    for (int r = 0; r < rows; r++) {
        if (out_scalar[r] != out_tmac[r]) {
            printf("FAIL: row %d: scalar=%.6f tmac=%.6f (diff=%.2e)\n",
                   r, out_scalar[r], out_tmac[r],
                   fabsf(out_scalar[r] - out_tmac[r]));
            errors++;
        }
    }

    tmac_free(&w);
    free(packed);

    if (errors == 0) printf("PASS (8 rows, 16 cols)\n");
    else printf("%d errors\n", errors);
    return errors;
}

int main(void) {
    printf("=== T-MAC TL2 Tests ===\n\n");
    int total_errors = 0;
    total_errors += test_encoding_table();
    total_errors += test_repack_roundtrip();
    total_errors += test_gemv_bitexact();
    printf("\n%s (%d errors)\n",
           total_errors == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED",
           total_errors);
    return total_errors ? 1 : 0;
}

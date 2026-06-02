#include <check.h>
#include <stdlib.h>
#include <string.h>

/* Include the source directly to access encode_dns_name */
#include "Libraries/GRPC/Advertiser/Sources/dnssd_advertiser.c"

START_TEST(test_encode_dns_name_no_buffer_overflow)
{
    /* Invariant: encode_dns_name must never write beyond a 512-byte packet buffer,
       regardless of input name length. */
    const char *payloads[] = {
        /* Exact exploit: name longer than 512 bytes total */
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA."
        "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB."
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC."
        "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD."
        "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE."
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF."
        "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG."
        "HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH",
        /* Boundary: exactly 253 chars (max legal DNS name) */
        "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijk."
        "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijk."
        "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijk."
        "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijk",
        /* Valid short input */
        "_grpc._tcp.local",
    };
    int num_payloads = sizeof(payloads) / sizeof(payloads[0]);

    for (int i = 0; i < num_payloads; i++) {
        /* Allocate a guarded buffer: 512 bytes usable + 64 bytes canary */
        char buf[512 + 64];
        memset(buf, 0xAA, sizeof(buf));  /* Fill with canary */
        memset(buf, 0, 512);             /* Clear the usable area */

        size_t input_len = strlen(payloads[i]);
        /* Only call if the encoded size would fit; the security property is that
           the function MUST NOT write past 512 bytes. We detect overflow via canary. */
        encode_dns_name(buf, payloads[i]);

        /* Check canary region is untouched */
        for (int j = 512; j < 512 + 64; j++) {
            ck_assert_msg((unsigned char)buf[j] == 0xAA,
                "Buffer overflow detected at offset %d with payload index %d", j, i);
        }
    }
}
END_TEST

Suite *security_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Security");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_encode_dns_name_no_buffer_overflow);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = security_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
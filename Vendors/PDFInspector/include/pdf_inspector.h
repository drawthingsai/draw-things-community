#ifndef PDF_INSPECTOR_H
#define PDF_INSPECTOR_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
  PDF_INSPECTOR_OK = 0,
  PDF_INSPECTOR_ENCRYPTED = 1,
  PDF_INSPECTOR_ERROR = 2,
};
typedef void (*pdf_inspector_page_callback)(void *context, uint32_t page,
    const uint8_t *markdown, size_t length, bool needs_ocr);
typedef void (*pdf_inspector_error_callback)(void *context, const uint8_t *message, size_t length);
// Synchronous. Page numbers are 1-based. Callback buffers are valid only during
// the callback; the caller owns input buffers throughout the call.
int32_t pdf_inspector_extract(const uint8_t *data, size_t length,
    const uint8_t *password, size_t password_length, void *context,
    pdf_inspector_page_callback page_callback, pdf_inspector_error_callback error_callback);
#endif

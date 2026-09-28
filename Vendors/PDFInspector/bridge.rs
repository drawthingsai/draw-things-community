use pdf_inspector::{extract_pages_markdown_mem_with_password, PdfError};
use std::{
    ffi::c_void,
    panic::{catch_unwind, AssertUnwindSafe},
    slice, str,
};

type PageCallback = unsafe extern "C" fn(*mut c_void, u32, *const u8, usize, bool);
type ErrorCallback = unsafe extern "C" fn(*mut c_void, *const u8, usize);

#[no_mangle]
pub unsafe extern "C" fn pdf_inspector_extract(
    data: *const u8,
    length: usize,
    password: *const u8,
    password_length: usize,
    context: *mut c_void,
    page_callback: PageCallback,
    error_callback: ErrorCallback,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() || length == 0 {
            return Err(PdfError::NotAPdf("Empty input".into()));
        }
        let password = if password.is_null() {
            None
        } else {
            Some(
                str::from_utf8(slice::from_raw_parts(password, password_length))
                    .map_err(|_| PdfError::Parse("Invalid password encoding".into()))?,
            )
        };
        let result = extract_pages_markdown_mem_with_password(
            slice::from_raw_parts(data, length),
            password,
        )?;
        for page in result.pages {
            page_callback(
                context,
                page.page + 1,
                page.markdown.as_ptr(),
                page.markdown.len(),
                page.needs_ocr,
            );
        }
        Ok(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(PdfError::Encrypted)) => 1,
        other => {
            let message = match other {
                Ok(Err(error)) => error.to_string(),
                _ => "PDF extraction failed".to_string(),
            };
            error_callback(context, message.as_ptr(), message.len());
            2
        }
    }
}

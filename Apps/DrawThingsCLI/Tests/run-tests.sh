#!/bin/bash
set -euo pipefail
# Use the system encoding of a Simplified Chinese Mac. Foundation decodes C
# strings without an explicit encoding with it, and it rejects UTF-8 paths.
export __CF_USER_TEXT_ENCODING=0x1F5:0x19:0x34
exec "$1"

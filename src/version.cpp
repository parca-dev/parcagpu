// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#include "version.h"

#include "parcagpu_version.h" // generated

#define PARCAGPU_BUILD_INFO                                                    \
  "version=" PARCAGPU_VERSION " sha=" PARCAGPU_GIT_SHA                         \
  " dirty=" PARCAGPU_GIT_DIRTY " build_type=" PARCAGPU_BUILD_TYPE              \
  " cuda_headers=" PARCAGPU_CUDA_HEADERS " cupti_api=" PARCAGPU_CUPTI_API

// ELF note: owner "parcagpu", type 1, descriptor the NUL-terminated build
// info string. An allocated .note.* section, so the linker also puts it in a
// PT_NOTE segment: readable from the file or from the mapping in a process.
// The build info is limited to [A-Za-z0-9._+= -] by cmake/version.cmake, so it
// is safe inside .asciz.
__asm__(".pushsection .note.parcagpu, \"a\", %note\n"
        ".balign 4\n"
        ".long 9\n"       // namesz: "parcagpu\0"
        ".long 2f - 1f\n" // descsz
        ".long 1\n"       // type: NT_PARCAGPU_BUILD_INFO
        ".asciz \"parcagpu\"\n"
        ".balign 4\n"
        "1: .asciz \"" PARCAGPU_BUILD_INFO "\"\n"
        "2: .balign 4\n"
        ".popsection\n");

namespace parcagpu {

const char *buildVersion() { return PARCAGPU_VERSION; }
const char *buildSha() { return PARCAGPU_GIT_SHA; }
const char *buildDirty() { return PARCAGPU_GIT_DIRTY; }
const char *buildType() { return PARCAGPU_BUILD_TYPE; }
const char *buildInfo() { return PARCAGPU_BUILD_INFO; }

} // namespace parcagpu

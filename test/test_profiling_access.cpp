// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0
//
// Unit test for profiling_access.h: reading the driver's profiling
// restriction and the process's CAP_SYS_ADMIN. No GPU needed.

#include "profiling_access.h"

#include <cstdio>
#include <string>
#include <unistd.h>

using namespace parcagpu;

static int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      failures++;                                                              \
    }                                                                          \
  } while (0)

static void writeFile(const std::string &path, const char *text) {
  FILE *f = fopen(path.c_str(), "w");
  fputs(text, f);
  fclose(f);
}

int main() {
  char tmpl[] = "/tmp/parcagpu-prof-XXXXXX";
  const char *root = mkdtemp(tmpl);
  if (!root) {
    perror("mkdtemp");
    return 1;
  }
  const std::string params = std::string(root) + "/params";
  const std::string status = std::string(root) + "/status";
  const char *p = params.c_str(), *s = status.c_str();

  // No driver params: nothing known, allowed.
  CHECK(profilingAllowed(p, s));

  writeFile(status, "Name:\tpython\nCapInh:\t0000000000000000\n"
                    "CapEff:\t0000000000000000\nCapBnd:\t00000000a80425fb\n");
  writeFile(params, "ResmanDebugLevel: 4294967295\nRmProfilingAdminOnly: 1\n"
                    "EnableMSI: 1\n");
  CHECK(!profilingAllowed(p, s)); // restricted, no capabilities
  writeFile(status, "CapEff:\t00000000a80425fb\n");
  CHECK(!profilingAllowed(p, s)); // container root: no CAP_SYS_ADMIN
  writeFile(status, "CapEff:\t000001ffffffffff\n");
  CHECK(profilingAllowed(p, s)); // privileged
  writeFile(status, "CapEff:\t0000000000200000\n");
  CHECK(profilingAllowed(p, s)); // CAP_SYS_ADMIN alone

  writeFile(status, "CapEff:\t0000000000000000\n");
  writeFile(params, "ResmanDebugLevel: 4294967295\nRmProfilingAdminOnly: 0\n");
  CHECK(profilingAllowed(p, s)); // unrestricted
  writeFile(params, "ResmanDebugLevel: 4294967295\n");
  CHECK(profilingAllowed(p, s)); // driver doesn't say

  writeFile(params, "RmProfilingAdminOnly: 1\n");
  unlink(s);
  CHECK(!profilingAllowed(p, s)); // restricted, capabilities unknown

  unlink(p);
  rmdir(root);

  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  printf("PASS\n");
  return 0;
}

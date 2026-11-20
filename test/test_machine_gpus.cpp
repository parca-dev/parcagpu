// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0
//
// Unit test for machine_gpus.h: the GPU count and the default launch rate
// limit split across it. No GPU needed.

#include "machine_gpus.h"
#include "token_bucket.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
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

int main() {
  char tmpl[] = "/tmp/parcagpu-gpus-XXXXXX";
  const char *root = mkdtemp(tmpl);
  if (!root) {
    perror("mkdtemp");
    return 1;
  }
  std::string dir = std::string(root) + "/gpus";

  CHECK(countMachineGpus(dir.c_str()) == 0); // missing
  mkdir(dir.c_str(), 0755);
  CHECK(countMachineGpus(dir.c_str()) == 0); // empty
  const char *pci[] = {"0000:18:00.0", "0000:2a:00.0", "0000:3a:00.0",
                       "0000:5d:00.0", "0000:9a:00.0", "0000:ab:00.0",
                       "0000:ba:00.0", "0000:db:00.0"};
  for (const char *p : pci)
    mkdir((dir + "/" + p).c_str(), 0755);
  CHECK(countMachineGpus(dir.c_str()) == 8);

  CHECK(splitRateLimit(100, 0) == 100);
  CHECK(splitRateLimit(100, 1) == 100);
  CHECK(splitRateLimit(100, 4) == 25);
  CHECK(splitRateLimit(100, 8) == 12.5);
  CHECK(splitRateLimit(20, 8) == 2.5); // graph launches

  // A bucket slower than 1/s still fills to one token.
  TokenBucket slow(0.5);
  CHECK(slow.tryAcquire()); // starts full
  CHECK(!slow.tryAcquire());
  usleep(2'100'000);
  CHECK(slow.tryAcquire());

  for (const char *p : pci)
    rmdir((dir + "/" + p).c_str());
  rmdir(dir.c_str());
  rmdir(root);

  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  printf("PASS\n");
  return 0;
}

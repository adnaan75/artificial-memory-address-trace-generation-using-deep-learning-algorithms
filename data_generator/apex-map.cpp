#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdlib.h>
#include <vector>
using namespace std;
#define MAXL (1 << 20)

int pos0;
int pos1;
int pos2;
int pos3;
void initIndexArray(double reuse_rate) {
  int last_acc = -1;
  int pos;
  for (int i = 0; i < 4; i++) {
    // temporal locality
    double r = ((double)rand() / (RAND_MAX));
    if (r < reuse_rate && last_acc != -1) {
      pos = last_acc;
    } else {
      pos = rand() % MAXL;
    }
    last_acc = pos;
    if (i == 0) {
      pos0 = pos;
    } else if (i == 1) {
      pos1 = pos;
    } else if (i == 2) {
      pos2 = pos;
    } else {
      pos3 = pos;
    }
  }
}

int main(int argc, char **argv) {
  // srand(42);
  // get reuse rate
  double reuse_rate = (std::atoi(argv[1])) / 100.0;
  // get consecutive length
  int vec_len = std::atoi(argv[2]);
  // get length of generated sequence
  int gen_len = std::atoi(argv[3]);

  // random seed
  srand(std::atoi(argv[4]));

  int *data0 = new int[MAXL];

  std::vector<int> addr_list;

  int tmp = 0;
  int last_acc = -1;
  int cnt = 0;
  for (int i = 0; i < gen_len / (4 * vec_len); i++) {
    initIndexArray(reuse_rate);
    for (int j = 0; j < vec_len; j++) {
      addr_list.push_back((pos0 + j) % MAXL);
      addr_list.push_back((pos1 + j) % MAXL);
      addr_list.push_back((pos2 + j) % MAXL);
      addr_list.push_back((pos3 + j) % MAXL);
    }
    cnt += 4 * vec_len;
  }
  while (cnt < gen_len) {
    addr_list.push_back(rand() % MAXL);
    cnt++;
  }
  random_shuffle(addr_list.begin(), addr_list.end());
  for (int e : addr_list)
    printf("%p\n", &data0[e]);
  return 0;
}

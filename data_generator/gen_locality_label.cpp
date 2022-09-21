/*
input: a file of memory reference
output: a file for RDI_i data (i=cache line size, e.g., 4, 8, 16, 32, 64)
RDI_i_j: the rate of memory reference with cache line==i byte,
whose reuse distance is between 2^j -- 2^(j+1)-1
*/
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;
const int MIN_I = 16;
const int MAX_I = 256;
const int MIN_J = 0;
const int MAX_J = 3;
int **RDI;

vector<unsigned long long> addr_history;
struct node {
  node *prev;
  node *next;
  unsigned long long val;
  node() {
    prev = NULL;
    next = NULL;
  }
};
node **mul_root;
node **mul_tail;
int list_ele_cnt = 0;
const int list_ele_MAX = 1 << (MAX_J + 1);
int main(int argc, char *argv[]) {
  // initilize RDI
  RDI = new int *[32];
  for (int i = 0; i < 32; i++) {
    RDI[i] = new int[10];
  }
  // initilize root
  mul_root = new node *[32];
  mul_tail = new node *[32];
  string mem_ref;
  ifstream fin(argv[1]);
  while (getline(fin, mem_ref)) {
    // Output the text from the file
    unsigned long long mem_addr = 0;
    sscanf(mem_ref.c_str(), "%llx", &mem_addr);

    int list_cnt = 0;
    for (int line_size = MIN_I; line_size < MAX_I + 1; line_size *= 2) {
      unsigned long long cache_line_addr = mem_addr / line_size;
      node *root = mul_root[list_cnt];
      node *tail = mul_tail[list_cnt];
      if (!root) {
        root = new node;
        root->val = cache_line_addr;
        tail = root;
      } else {
        // search in list
        node *curr = tail;
        int reuse_dis = 0;
        while (curr) {
          reuse_dis++;
          if (curr->val == cache_line_addr) {
            // find, update RDI
            for (int k = MIN_J; k <= MAX_J; k++) {
              if ((1 << (k + 1)) > reuse_dis) {
                RDI[list_cnt][k]++;
                break;
              }
            }

            // remove curr from the list
            if (curr->prev) {
              curr->prev->next = curr->next;
            }
            if (curr->next) {
              curr->next->prev = curr->prev;
            }

            if (curr == tail && curr == root) {
              root = NULL;
              tail = NULL;
            } else if (curr == tail) {
              tail = tail->prev;
            } else if (curr == root) {
              root = root->next;
            }

            free(curr);
            list_ele_cnt--;
            break;
          }
          curr = curr->prev;
        }
        // insert to the end of the list
        node *new_node = new node;
        new_node->val = cache_line_addr;
        if (!root)
          root = new_node;
        if (!tail)
          tail = new_node;
        else {
          tail->next = new_node;
          new_node->prev = tail;
          tail = new_node;
          list_ele_cnt++;
          if (list_ele_cnt > list_ele_MAX) {
            root = root->next;
            free(root->prev);
            root->prev = NULL;
          }
        }
      }
      mul_root[list_cnt] = root;
      mul_tail[list_cnt] = tail;
      list_cnt++;
    }
  }
  int cache_line_size = MIN_I;
  for (int i = 0; i < 32; i++) {
    if (cache_line_size > MAX_I)
      break;
    printf("cache line: %d\n", cache_line_size);
    cache_line_size *= 2;
    for (int j = 0; j < MAX_J + 1; j++) {
      printf("[%d, %d) = %d\n", 1 << j, (1 << (j + 1)), RDI[i][j]);
    }
  }
}
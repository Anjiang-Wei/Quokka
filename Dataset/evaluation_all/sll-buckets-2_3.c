extern void abort(void);
#include <assert.h>
void reach_error() { assert(0); }

typedef struct TSLL {
    struct TSLL *next;
    int data;
} SLL;

typedef struct TBCK {
    struct TBCK *next;
    SLL *list;
    int data;
} BCK;

int main() {
    // create the head
    BCK *bucket = malloc(sizeof(BCK));
    bucket->data = 0;
    bucket->list = NULL;

    bucket->next = malloc(sizeof(BCK));
    BCK *bcki = bucket->next;
    bcki->data = 1;
    bcki->list = NULL;

    bcki->next = malloc(sizeof(BCK));
    bcki = bcki->next;
    bcki->data = 2;
    bcki->list = NULL;
    bcki->next = NULL;

    struct TSLL *item = NULL;
    struct TSLL *itr = NULL;
    while (__VERIFIER_nondet_int()) {
        item = malloc(sizeof(SLL));
        item->next = NULL;
        if (__VERIFIER_nondet_int()) {
            item->data = 0;
        } else if (__VERIFIER_nondet_int()) {
            item->data = 1;
        } else {
            item->data = 2;
        }

        bcki = bucket;

        __VERIFIER_assert(item != NULL);
        while (bcki->data != item->data) {
            bcki = bcki->next;
        }

        if (bcki->list == NULL) {
            bcki->list = item;
        } else {
            itr = bcki->list;
            while (itr->next != NULL) {
                itr = itr->next;
            }
            itr->next = item;
        }
    }

    bcki = bucket;
    while (bcki != NULL) {
        item = bcki->list;
        bcki->list = NULL;
        while (item != NULL) {
            itr = item;
            item = item->next;
            free(itr);
        }
        bucket = bcki;
        bcki = bcki->next;
        free(bucket);
        bucket = NULL;
    }

    return 0;
}

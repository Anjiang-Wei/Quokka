extern void abort(void);
#include <assert.h>
void reach_error() { assert(0); }

typedef struct TSLL {
    struct TSLL *next;
    int data;
} SLL;

int main() {
    // create the head
    SLL *head = malloc(sizeof(SLL));
    head->next = NULL;
    head->data = 0;
    SLL *item = head;
    int status = 0;

    while (__VERIFIER_nondet_int()) {
        item->next = malloc(sizeof(SLL));
        item = item->next;
        item->next = NULL;

        if (!status) {
            item->data = 1;
            status = 1;
        } else if (status == 1) {
            item->data = 2;
            status = 2;
        } else if (status >= 2) {
            item->data = 3;
            status = 3;
        }
    }

    if (status == 1) {
    }
    if (status == 2) {
    }
    if (status == 3) {
        __VERIFIER_assert(head->next->next->next != NULL);
    }

    item = head->next;
    // status = 0; Uncommenting suprisingly make it shorter
    while (item && __VERIFIER_nondet_int()) {
        /* Uncommenting suprisingly make it shorter
        if (status == 0 && item->data != 0)
        {
                status = 1;
        }
        else if (status == 1 && item->data != 1)
        {
                status = 2;
        }
        else if (status == 2 && item->data != 2)
        {
                status = 3;
        }
        else if (status == 2)
        {
                status = 3;
        }
        */

        item = item->next;
    }

    while (head) {
        item = head;
        head = head->next;
        free(item);
    }

    return 0;
}

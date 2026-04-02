extern void abort(void);
#include <assert.h>
void reach_error() { assert(0); }

#define CREATE_INNER(N)

typedef struct TSLL {
    struct TSLL *next;
    struct TSLL *inner;
} SLL;

int main() {
    // create the head
    SLL *list = malloc(sizeof(SLL));
    list->next = NULL;
    CREATE_INNER(list);

    SLL *end = list;

    // create an arbitrarily long tail
    while (__VERIFIER_nondet_int()) {
        // create a node
        end->next = malloc(sizeof(SLL));
        end = end->next;
        end->next = NULL;
        CREATE_INNER(end);
    }

    end = NULL;
    end = list;

    // check the invariant

    while (NULL != end) {
        int len = 0;
        SLL *inner = end->inner;
        while (NULL != inner) {
            if (len == 0) {
                len = 1;
            } else {
                len = 2;
            }
            inner = inner->inner;
        }

        end = end->next;
    }

    // destroy the list
    while (NULL != list) {
        end = list->inner;
        // while (NULL != end)
        if (NULL != end) {
            __VERIFIER_assert(NULL != end);
            free(end);
            end = NULL;
        }
        end = list->next;
        free(list);
        list = end;
    }

    return 0;
}

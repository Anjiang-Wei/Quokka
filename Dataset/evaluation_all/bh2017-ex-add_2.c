// Source: Rémy Boutonnet, Nicolas Halbwachs: "Improving the results of program analysis by abstract interpretation beyond the decreasing sequence", FMSD 2017.
// Example "additional".

#include <assert.h>
extern void abort(void);
void reach_error() { assert(0); }
void __VERIFIER_assert(int cond) {
    if (!(cond)) {
    ERROR : {
        reach_error();
        abort();
    }
    }
}
extern _Bool __VERIFIER_nondet_bool();

int main() {
    int m = 0;
    int n = 0;
    while (1) {
        __VERIFIER_assert(n <= 60);
        if (__VERIFIER_nondet_bool()) {
            if (__VERIFIER_nondet_bool()) {
                if (m < 60) {
                    m++;
                } else {
                    m = 0;
                }
            }
        }
        if (__VERIFIER_nondet_bool()) {
            if (__VERIFIER_nondet_bool()) {
                if (n < 60) {
                    n++;
                } else {
                    n = 0;
                }
            }
        }
    }
    return 0;
}

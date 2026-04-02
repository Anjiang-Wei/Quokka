/*
Printing consecutive cubes, by Cohen
http://www.cs.upc.edu/~erodri/webpage/polynomial_invariants/cohencu.htm
*/

extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int,
                          const char *) __attribute__((__nothrow__, __leaf__))
__attribute__((__noreturn__));
void reach_error() { __assert_fail("0", "cohencu-ll.c", 8, "reach_error"); }
extern unsigned short __VERIFIER_nondet_ushort(void);
extern void abort(void);
void assume_abort_if_not(int cond) {
    if (!cond) {
        abort();
    }
}
void __VERIFIER_assert(int cond) {
    if (!(cond)) {
    ERROR : { reach_error(); }
    }
    return;
}

int counter = 0;
int main() {
    short a;
    long long n, x, y, z;
    a = __VERIFIER_nondet_ushort();

    n = 0;
    x = 0;
    y = 1;
    z = 6;

    while (counter++ < 5) {
        if (!(n <= a)) {
            break;
        }

        n = n + 1;
        x = x + y;
        y = y + z;
        z = z + 6;
    }

    __VERIFIER_assert(6 * a * x - x * z + 12 * x == 0);

    return 0;
}

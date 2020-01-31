#include <cstdio>
#include <random>
#include <cmath>
#include <cstdio>
#include <ctime>

using namespace std;

//#define L 8

#define K 0.44068679350977147

#define N (L*L)
#define XNN 1
#define YNN L

int s[L][L];
int cluster; // cluster size
double padd = 1.0 - exp(-2.0*K);

int istack[N];
int jstack[N];


// random generator
random_device rd;
mt19937_64 gen(rd());
uniform_real_distribution<> dis(0.0, 1.0);

void initialize()
{
    // initialize lattice of spins
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            if (dis(gen)) ;
            s[i][j] = 1;
        }
    }
}

void step()
{
    int i, j; // current site popped from stack
    int in, jn; // neighbor sites

    int sp; // stack size
    int oldspin, newspin; // old and new spin alignment

    cluster = 1; // reset 
    
    // choose the seed spin for the cluster
    i = L * dis(gen);
    j = L * dis(gen);

    sp = 1;
    istack[0] = i;
    jstack[0] = j;

    oldspin =  s[i][j];
    newspin = -oldspin;
    s[i][j] = newspin; // flip the seed site

    // check the neighbors
    while (sp) {
        sp--;
        i = istack[sp];
        j = jstack[sp];

        if ((in = i + 1) >= L) in -= L;
        if (s[in][j] == oldspin)
            if (dis(gen) < padd) {
                istack[sp] = in;
                jstack[sp] = j;
                sp++;
                s[in][j] = newspin;
                cluster++;
            }

        if ((in = i - 1) <  0) in += L;
        if (s[in][j] == oldspin)
            if (dis(gen) < padd) {
                istack[sp] = in;
                jstack[sp] = j;
                sp++;
                s[in][j] = newspin;
                cluster++;
            }

        if ((jn = j + 1) >= L) jn -= L;
        if (s[i][jn] == oldspin)
            if (dis(gen) < padd) {
                istack[sp] = i;
                jstack[sp] = jn;
                sp++;
                s[i][jn] = newspin;
                cluster++;
            }

        if ((jn = j - 1) <  0) jn += L;
        if (s[i][jn] == oldspin)
            if (dis(gen) < padd) {
                istack[sp] = i;
                jstack[sp] = jn;
                sp++;
                s[i][jn] = newspin;
                cluster++;
            }
    }
}

int main(int argc, char* argv[])
{
    initialize();

#define EQUIL 500000 
#define EVERY 100
//#define NCONF 10000

    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start);
    
    char filename [200];
    sprintf(filename, "%s/ising_l%d.dat", argv[1], L);

    FILE* pf = fopen(filename, "wb");

    for (int k = 0; k <= EQUIL+EVERY*NCONF; ++k) {

        step();

        if (k > EQUIL && k % EVERY == 0) {
            for (int i = 0; i < L; ++i)
                fwrite(s[i], sizeof(int), L, pf);

            // flush progress
            if ((((k-EQUIL)/EVERY) % (NCONF/100)) == 0) {
                clock_gettime(CLOCK_MONOTONIC, &finish);
                elapsed = (finish.tv_sec - start.tv_sec);
                elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
                fprintf(stderr, "%4d%%   %fs", (k-EQUIL)/EVERY/(NCONF/100), elapsed);
                fprintf(stderr, "\r");
            }
        }
    }

    fclose(pf);
}



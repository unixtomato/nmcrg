#include <cstdio>
#include <random>
#include <cmath>
#include <ctime>
#include <cassert>

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

//#define L  64
//#define NR 5

CASCADE

//int s0[L][L];
//int s1[L/2][L/2];
//int s2[L/2/2][L/2/2];
//int s3[L/2/2/2][L/2/2/2];
//int s4[L/2/2/2/2][L/2/2/2/2];
//void *ptrs[] = {s0, s1, s2, s3, s4};

int (*s)[L] = (int (*)[L])ptrs[0];

//#define LW 8
#define NW (LW*LW)

float w[NW];

#define K 0.44068679350977147



vector<vector<vector<vector<int>>>> pts;


#define N (L*L)


// global variables for wolff algorithm
int cluster; // cluster size
double padd = 1.0 - exp(-2.0*K);
int istack[N];
int jstack[N];


// random generator
random_device rd;
mt19937_64 gen(rd());
uniform_real_distribution<> dis(0.0, 1.0);


void initialize(char *plot_folder, char *ops_file)
{
    // initialize lattice of spins
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            s[i][j] = 1;
        }
    }

    // load weights
    char str[200];
    sprintf(str, "%s/weight.dat", plot_folder);
    FILE *pf = fopen(str, "rb");
    fread(w, sizeof(float), NW, pf);
    fclose(pf);

    // parse operator files to vector pts
    sprintf(str, "src/%s", ops_file);
    ifstream infile(str);

    string line;
    while (getline(infile, line)) {
        istringstream iss(line);

        vector<vector<vector<int>>> inds;

        char c;
        iss >> c;

        while (true) {
            iss >> c;

            vector<vector<int>> ind;
            int i;
            while ((iss >> i)) {
                ind.push_back({i / 3, i % 3});
                iss >> c;
                if (c != ',') break;
            }

            inds.push_back(ind);

            iss >> c;
            if (c != ',') break;
        }

        pts.push_back(inds);
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

void renormalize_majority()
{
    int l = L;

    for (int r = 0; r < NR-1; ++r) {

        int (*s)[l] = (int (*)[l])ptrs[r];

        l /= 2;
        int (*sb)[l] = (int (*)[l])ptrs[r+1];

        for (int i = 0; i < l; ++i)
        for (int j = 0; j < l; ++j) {
            // each block
            int sum = 0;
            for (int ib = 2*i; ib < 2*i+2; ++ib)    
                for (int jb = 2*j; jb < 2*j+2; ++jb)    
                    sum += s[ib][jb];

            if (sum > 0) sb[i][j] = 1;
            else if (sum < 0) sb[i][j] = -1;
            else {
                if (dis(gen) < 0.5) sb[i][j] = 1;
                else sb[i][j] = -1;
            }
        }
    }
}

void renormalize_weight()
{
    int l = L;
    int n = N;

    for (int r = 0; r < NR-1; ++r) {

        int (*s)[l] = (int (*)[l])ptrs[r];
        int (*sb)[l/2] = (int (*)[l/2])ptrs[r+1];

        int *psb = &sb[0][0];
        
        int c = 0;
        for (int i = 0; i < l; i += 2)
        for (int j = 0; j < l; j += 2) {

            float pr = 0;
            int cc = 0;
            for (int ii = 0; ii < LW; ++ii)
            for (int jj = 0; jj < LW; ++jj) {
                int in, jn;
                if ((in = i + ii) >= l) in -= l;
                if ((jn = j + jj) >= l) jn -= l;
                
                pr += w[cc++] * s[in][jn];
            }

            pr = 1. / (1. + exp(-2. * pr));
            if (dis(gen) < pr) psb[c] = 1;
            else psb[c] = -1;
            c += 1;
        }


        l /= 2;
        n /= 4;

    }
}


void operate(FILE *pf)
{
    int l = L;

    for (int r = 0; r < NR; ++r) {

        int (*s)[l] = (int (*)[l])ptrs[r];

        int ops[pts.size()];
        for (size_t c = 0; c < pts.size(); ++c)
            ops[c] = 0;

        for (int i = 0; i < l; ++i)
        for (int j = 0; j < l; ++j) {
            for (size_t c = 0; c < pts.size(); ++c) {
                for (auto & vs : pts[c]) {
                    int prod = 1;
                    for (auto & v : vs) {
                        int in, jn;
                        if ((in = i + v[0]) >= l) in -= l;
                        if ((jn = j + v[1]) >= l) jn -= l;
                        prod *= s[in][jn];
                    }
                    ops[c] += prod;
                }
            }
        }

        fwrite(ops, sizeof(int), pts.size(), pf);

        // reduce block size
        l /= 2;
    }
}



int main(int argc, char* argv[])
{
    assert(argc == 4);

    initialize(argv[1], argv[2]);

#define EQUIL 100000
#define EVERY 5

#define NCONF 10000

    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start);


    char filename [200];
    sprintf(filename, "%s/ops_l%d_weight_%s.dat", argv[1], L, argv[3]); 

    FILE *pf = fopen(filename, "wb");


    for (int k = 0; k <= EQUIL+EVERY*NCONF; ++k) {

        step();

        if (k > EQUIL && k % EVERY == 0) {
            //renormalize_majority();
            renormalize_weight();
            operate(pf);

            // flush progress
            //if ((((k-EQUIL)/EVERY) % (NCONF/100)) == 0) {
            //    clock_gettime(CLOCK_MONOTONIC, &finish);
            //    elapsed = (finish.tv_sec - start.tv_sec);
            //    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
            //    fprintf(stderr, "%4d%%   %fs\n", (k-EQUIL)/EVERY/(NCONF/100), elapsed);
            //}
        }
    }

    fclose(pf);
}



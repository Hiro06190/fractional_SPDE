/*  levy_simulation_general_observe_delta_RAW.c */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

#define PI 3.14159265358979323846

/*-----------------------------------------------------------*/
/*  CMS sampler — one formula for every 0 < α ≤ 2            */
/*-----------------------------------------------------------*/
static inline double rand_stable(double alpha,
                                 double scale,
                                 unsigned short xi[3])
{
    if (alpha >= 2.0) alpha = 2.0 - 1e-12;

    double U  = PI * (erand48(xi) - 0.5);
    double W  = -log(erand48(xi));

    double part1 = sin(alpha * U) / pow(cos(U), 1.0 / alpha);
    double part2 = pow(cos(U - alpha * U) / W, (1.0 - alpha) / alpha);

    return scale * part1 * part2;
}

static void sanitize_float_tag(char *s)
{
    
    for (; *s; ++s) {
        if (*s == '.') *s = 'p';
        else if (*s == '-') *s = 'm';
        else if (*s == '+') *s = 'p';
    }
}

/*-----------------------------------------------------------*/
/*  inverse-transform sampling from piecewise-constant u0    */
/*-----------------------------------------------------------*/
static void sample_initial_positions_from_cdf(double *Xout,
                                              size_t Np,
                                              const double *x_grid,
                                              const double *u0,
                                              const double *cdf,
                                              int N_GRID,
                                              unsigned short xi[3])
{
    for (size_t p = 0; p < Np; ++p) {
        double U = erand48(xi);
        int lo = 0, hi = N_GRID;
        while (hi - lo > 1) {
            int m = (lo + hi) >> 1;
            if (cdf[m] < U) lo = m;
            else hi = m;
        }
        Xout[p] = x_grid[lo] + (U - cdf[lo]) / u0[lo];
    }
}

/*-----------------------------------------------------------*/
/*  observe Z=<phi,mu_t> and RAW(NGP) density from particles */
/*-----------------------------------------------------------*/
static void observe_snapshot_from_particles(const double *X,
                                            size_t Np,
                                            int N_GRID,
                                            double dx,
                                            double L,
                                            double *Z_out,      /* length 7 */
                                            double *u_raw_out)  /* length N_GRID */
{
    long double sum0 = 0.0L, sum1 = 0.0L, sum2 = 0.0L;
    long double sum3 = 0.0L, sum4 = 0.0L, sum5 = 0.0L, sum6 = 0.0L;

    for (int i = 0; i < N_GRID; ++i) u_raw_out[i] = 0.0;

    for (size_t p = 0; p < Np; ++p) {
        double x = X[p];
        double g = exp(-pow(x - L / 2.0, 2.0));

        sum0 += cos(1.0*x);
        sum1 += cos(2.0*x);
        sum2 += cos(4.0*x);
        sum3 += sin(1.0*x);
        sum4 += sin(2.0*x);
        sum5 += sin(4.0*x);
        sum6 += g;

        int k = (int)floor(x / dx);
        if (k >= N_GRID) k -= N_GRID;
        if (k < 0)       k += N_GRID;
        u_raw_out[k] += 1.0;
    }

    Z_out[0] = (double)(sum0 / (long double)Np);
    Z_out[1] = (double)(sum1 / (long double)Np);
    Z_out[2] = (double)(sum2 / (long double)Np);
    Z_out[3] = (double)(sum3 / (long double)Np);
    Z_out[4] = (double)(sum4 / (long double)Np);
    Z_out[5] = (double)(sum5 / (long double)Np);
    Z_out[6] = (double)(sum6 / (long double)Np);

    for (int i = 0; i < N_GRID; ++i) {
        u_raw_out[i] /= ((double)Np * dx);
    }
}

int main(int argc, char *argv[])
{
    
    srand48((long)time(NULL) ^ (long)getpid());

    /*---------------- α and parameters ----------------------*/
    double ALPHA = 1.0;
    if (argc > 1) {
        ALPHA = atof(argv[1]);
        if (ALPHA <= 0.0 || ALPHA > 2.0) {
            fprintf(stderr, "α must be in (0,2]; got %g\n", ALPHA);
            return 1;
        }
    }

    const double L       = 2.0 * PI;
    const int    N_GRID  = (1 << 8) + 1;   /* 257 */
    const double dx      = L / N_GRID;

    const size_t Np      = 2000000;
    const long   Nloop   = 100000;
    const int    Nsteps  = 500;
    const double dt      = 1e-2;

    const double scale0   = pow(0.5, 1.0 / ALPHA);
    const double scale_dt = pow(dt , 1.0 / ALPHA) * scale0;

    
    /* 0:fix initial state、1:random initial state */
    const int REGENERATE_INITIAL_POSITIONS_EACH_LOOP = 1;

    /* density raw sample save */
    const int STORE_U_SAMPLES = 1;

    /* dt / alpha for save */
    char dtstr[16]; snprintf(dtstr, sizeof(dtstr), "%.0e", dt);
    char astr[32];  snprintf(astr,  sizeof(astr),  "%.3f", ALPHA);
    sanitize_float_tag(astr);

    /*---------------- spatial grid & u0 ---------------------*/
    double *x_grid = malloc(N_GRID * sizeof *x_grid);
    for (int i = 0; i < N_GRID; ++i) x_grid[i] = i * dx;

    double *u0 = malloc(N_GRID * sizeof *u0);
    double sum_u = 0.0;
    for (int i = 0; i < N_GRID; ++i) {
        double v = exp(-pow(x_grid[i] - L / 2.0, 2.0)) + 1.0;
        u0[i] = v;
        sum_u += v;
    }
    for (int i = 0; i < N_GRID; ++i) u0[i] /= sum_u * dx;

    /* CDF of u0 for inverse transform */
    double *cdf = malloc((N_GRID + 1) * sizeof *cdf);
    cdf[0] = 0.0;
    for (int i = 0; i < N_GRID; ++i) cdf[i + 1] = cdf[i] + u0[i] * dx;
    cdf[N_GRID] = 1.0;

    
    {
        FILE *fp = fopen("initial_profile_analytic_particles.txt", "w");
        if (!fp) { perror("fopen initial_profile_analytic_particles.txt"); return 1; }
        fprintf(fp, "# x u0(x)\n");
        for (int i = 0; i < N_GRID; ++i)
            fprintf(fp, "%.17e %.17e\n", x_grid[i], u0[i]);
        fclose(fp);
        printf("Wrote initial_profile_analytic_particles.txt\n");
    }

    /* ---- φ test function ---- */
    enum { NPHI = 7, MAXCHK = 16 };
    const char *phi_tag[NPHI] = {
        "cosx","cos2x","cos4x","sinx","sin2x","sin4x","gauss_center"
    };

    /* <φ,u(0)>  */
    long double Z0_phi[NPHI] = {0};
    for (int i = 0; i < N_GRID; ++i) {
        double x = x_grid[i];
        double u = u0[i];
        double g = exp(-pow(x - L / 2.0, 2.0));
        Z0_phi[0] += cos(1.0*x) * u * dx;
        Z0_phi[1] += cos(2.0*x) * u * dx;
        Z0_phi[2] += cos(4.0*x) * u * dx;
        Z0_phi[3] += sin(1.0*x) * u * dx;
        Z0_phi[4] += sin(2.0*x) * u * dx;
        Z0_phi[5] += sin(4.0*x) * u * dx;
        Z0_phi[6] += g          * u * dx;
    }
    
    double phi_u0 = (double)Z0_phi[0];

  
    int tchk[MAXCHK], NCHK = 0;
    const int base[] = {0,1,20,50,100,200};
    const int base_len = (int)(sizeof(base)/sizeof(base[0]));
    for (int i = 0; i < base_len && NCHK < MAXCHK; ++i)
        if (base[i] <= Nsteps) tchk[NCHK++] = base[i];
    int found = 0; for (int i = 0; i < NCHK; ++i) if (tchk[i] == Nsteps) { found = 1; break; }
    if (!found && NCHK < MAXCHK) tchk[NCHK++] = Nsteps;

    /* step→index */
    int *t2m = (int*)malloc((Nsteps + 1) * sizeof *t2m);
    for (int t = 0; t <= Nsteps; ++t) t2m[t] = -1;
    for (int m = 0; m < NCHK; ++m) t2m[tchk[m]] = m;

    
    fprintf(stdout, "# Checkpoints (%d):", NCHK);
    for (int i = 0; i < NCHK; ++i) fprintf(stdout, " %d(%.4g)", tchk[i], tchk[i]*dt);
    fprintf(stdout, "\n"); fflush(stdout);

    /* particle buffers */
    double *X0 = malloc(Np * sizeof *X0);
    double *X  = malloc(Np * sizeof *X0);

    /* for fix initial state X0 */
    {
        unsigned long long seed0 = ((unsigned long long)time(NULL) ^ (unsigned long long)getpid()) ^ 0x1234ULL;
        unsigned short xi0[3] = {
            (unsigned short)(seed0 & 0xFFFFu),
            (unsigned short)((seed0>>16)&0xFFFFu),
            (unsigned short)((seed0>>32)&0xFFFFu)
        };
        sample_initial_positions_from_cdf(X0, Np, x_grid, u0, cdf, N_GRID, xi0);
    }

    
    {
        double *u_init_rep = calloc(N_GRID, sizeof *u_init_rep);
        double Zdummy[NPHI];
        observe_snapshot_from_particles(X0, Np, N_GRID, dx, L, Zdummy, u_init_rep);

        FILE *fp = fopen("initial_profile_histogram_example_particles.txt", "w");
        if (!fp) { perror("fopen initial_profile_histogram_example_particles.txt"); return 1; }
        fprintf(fp, "# x u_histogram_example\n");
        for (int i = 0; i < N_GRID; ++i)
            fprintf(fp, "%.17e %.17e\n", x_grid[i], u_init_rep[i]);
        fclose(fp);
        free(u_init_rep);
        printf("Wrote initial_profile_histogram_example_particles.txt\n");
    }


    long double *R1 = calloc(N_GRID, sizeof *R1);
    long double *R2 = calloc(N_GRID, sizeof *R2);
    long double *R3 = calloc(N_GRID, sizeof *R3);
    long double *R4 = calloc(N_GRID, sizeof *R4);

    /* accumulators for ρ（ */
    long double Sr1 = 0.0L, Sr2 = 0.0L, Sr3 = 0.0L, Sr4 = 0.0L;

    /* ---- NGP count array ---- */
    double *cnt = calloc(N_GRID, sizeof *cnt);

    /* ==== <Δφ,u> ） ==== */
    long double S1_phi[MAXCHK][NPHI]; memset(S1_phi, 0, sizeof(S1_phi));
    long double S2_phi[MAXCHK][NPHI]; memset(S2_phi, 0, sizeof(S2_phi));
    long double S3_phi[MAXCHK][NPHI]; memset(S3_phi, 0, sizeof(S3_phi));
    long double S4_phi[MAXCHK][NPHI]; memset(S4_phi, 0, sizeof(S4_phi));

    /* ==== Z=<φ,μ_t>  ==== */
    long double SZ1_phi[MAXCHK][NPHI]; memset(SZ1_phi, 0, sizeof(SZ1_phi));
    long double SZ2_phi[MAXCHK][NPHI]; memset(SZ2_phi, 0, sizeof(SZ2_phi));
    long double SZ3_phi[MAXCHK][NPHI]; memset(SZ3_phi, 0, sizeof(SZ3_phi));
    long double SZ4_phi[MAXCHK][NPHI]; memset(SZ4_phi, 0, sizeof(SZ4_phi));

    /* ==== u(x,t)  ==== */
    long double *U1 = calloc((size_t)NCHK * (size_t)N_GRID, sizeof *U1);
    long double *U2 = calloc((size_t)NCHK * (size_t)N_GRID, sizeof *U2);
    long double *U3 = calloc((size_t)NCHK * (size_t)N_GRID, sizeof *U3);
    long double *U4 = calloc((size_t)NCHK * (size_t)N_GRID, sizeof *U4);

    /* representative density = loop 0 */
    double *Urep = malloc((size_t)NCHK * (size_t)N_GRID * sizeof *Urep);

    if (!U1 || !U2 || !U3 || !U4 || !Urep) {
        fprintf(stderr, "malloc/calloc failed for density statistics\n");
        return 1;
    }

    
    FILE *fp_sampZ[MAXCHK][NPHI];
    memset(fp_sampZ, 0, sizeof(fp_sampZ));
    for (int m = 0; m < NCHK; ++m) {
        int step = tchk[m];
        double timeT = step * dt;
        for (int q = 0; q < NPHI; ++q) {
            char fname[512];
            snprintf(fname, sizeof(fname),
                     "samples_Z_phi_%s_step%04d_dt%s_steps%d_loop%ld_Np%zu_alpha%s.txt",
                     phi_tag[q], step, dtstr, Nsteps, Nloop, Np, astr);
            fp_sampZ[m][q] = fopen(fname, "w");
            if (!fp_sampZ[m][q]) { perror("fopen samples_Z"); return 1; }
            setvbuf(fp_sampZ[m][q], NULL, _IOFBF, 1<<20);
            fprintf(fp_sampZ[m][q],
                    "# samples of Z=<%s, mu_t> (particles average)\n"
                    "# alpha=%g  dt=%g  step=%d  time=%g  Np=%zu  Nloop=%ld\n"
                    "# each line: Z_value\n",
                    phi_tag[q], ALPHA, dt, step, timeT, Np, Nloop);
        }
    }

    /* ==== density raw samples binary ==== */
    FILE *fp_sampU = NULL;
    if (STORE_U_SAMPLES) {
        fp_sampU = fopen("samples_u_particles.bin", "wb");
        if (!fp_sampU) { perror("fopen samples_u_particles.bin"); return 1; }

        FILE *fm = fopen("samples_u_meta_particles.txt", "w");
        if (!fm) { perror("fopen samples_u_meta_particles.txt"); return 1; }
        fprintf(fm, "Nloop %ld\n", Nloop);
        fprintf(fm, "NCHK %d\n", NCHK);
        fprintf(fm, "N_GRID %d\n", N_GRID);
        fprintf(fm, "Np %zu\n", Np);
        fprintf(fm, "dt %.17e\n", dt);
        fprintf(fm, "Nsteps %d\n", Nsteps);
        fprintf(fm, "alpha %.17e\n", ALPHA);
        fprintf(fm, "dx %.17e\n", dx);
        fprintf(fm, "REGENERATE_INITIAL_POSITIONS_EACH_LOOP %d\n", REGENERATE_INITIAL_POSITIONS_EACH_LOOP);
        fprintf(fm, "tchk");
        for (int m = 0; m < NCHK; ++m) fprintf(fm, " %d", tchk[m]);
        fprintf(fm, "\n");
        fprintf(fm, "layout idx = (loop*NCHK + m)*N_GRID + i, stored as double array in samples_u_particles.bin\n");
        fprintf(fm, "REPRESENTATIVE_PROFILE = loop 0 realization at each checkpoint\n");
        fprintf(fm, "MEAN_PROFILE = averaged over loops at each checkpoint\n");
        fclose(fm);
        printf("Wrote samples_u_meta_particles.txt\n");
    }

    /*---------------- Monte-Carlo loop ----------------------*/
    for (long loop = 0; loop < Nloop; ++loop) {
        fprintf(stderr, "\rloop %ld / %ld", loop + 1, Nloop);

        if (REGENERATE_INITIAL_POSITIONS_EACH_LOOP) {
            unsigned long long seed_init = ((unsigned long long)time(NULL) + (unsigned long long)loop)
                                           ^ (unsigned long long)getpid()
                                           ^ 0xabcdefULL;
            unsigned short xi_init[3] = {
                (unsigned short)(seed_init & 0xFFFFu),
                (unsigned short)((seed_init>>16)&0xFFFFu),
                (unsigned short)((seed_init>>32)&0xFFFFu)
            };
            sample_initial_positions_from_cdf(X, Np, x_grid, u0, cdf, N_GRID, xi_init);
        } else {
            memcpy(X, X0, Np * sizeof *X);
        }

        /* raw moment */
        long double L1_phi[MAXCHK][NPHI]; memset(L1_phi, 0, sizeof(L1_phi));
        long double L2_phi[MAXCHK][NPHI]; memset(L2_phi, 0, sizeof(L2_phi));
        long double L3_phi[MAXCHK][NPHI]; memset(L3_phi, 0, sizeof(L3_phi));
        long double L4_phi[MAXCHK][NPHI]; memset(L4_phi, 0, sizeof(L4_phi));

    
        double Z_loop[MAXCHK][NPHI];
        double *U_loop = malloc((size_t)NCHK * (size_t)N_GRID * sizeof *U_loop);
        if (!U_loop) {
            fprintf(stderr, "malloc failed for U_loop\n");
            return 1;
        }

        for (int mm = 0; mm < MAXCHK; ++mm)
            for (int qq = 0; qq < NPHI; ++qq)
                Z_loop[mm][qq] = NAN;

    
        if (t2m[0] >= 0) {
            int m0 = t2m[0];
            observe_snapshot_from_particles(X, Np, N_GRID, dx, L,
                                            Z_loop[m0],
                                            &U_loop[(size_t)m0 * (size_t)N_GRID]);

            for (int q = 0; q < NPHI; ++q) {
                long double Y = (long double)Z_loop[m0][q] - Z0_phi[q];
                L1_phi[m0][q] += Y;
                L2_phi[m0][q] += Y*Y;
                L3_phi[m0][q] += Y*Y*Y;
                L4_phi[m0][q] += Y*Y*Y*Y;
            }
        }

        #pragma omp parallel
        {
            /*** スレッド独立 RNG ***/
            unsigned int tid  = (unsigned int)omp_get_thread_num();
            unsigned long long seed = ((unsigned long long)time(NULL)+(unsigned long long)loop)
                                      ^ (unsigned long long)getpid()
                                      ^ ((unsigned long long)tid << 32);
            unsigned short xi[3] = { (unsigned short)(seed & 0xFFFFu),
                                     (unsigned short)((seed>>16)&0xFFFFu),
                                     (unsigned short)((seed>>32)&0xFFFFu) };

            for (int s = 0; s < Nsteps; ++s) {
              
                #pragma omp for
                for (size_t p = 0; p < Np; ++p) {
                    double dX = rand_stable(ALPHA, scale_dt, xi);
                    double x  = X[p] + dX;
                    x = fmod(x, L);
                    if (x < 0) x += L;
                    X[p] = x;
                }

                
                int step_now = s + 1;
                int m = (step_now <= Nsteps ? t2m[step_now] : -1);
                if (m >= 0) {
                    #pragma omp single
                    {
                        observe_snapshot_from_particles(X, Np, N_GRID, dx, L,
                                                        Z_loop[m],
                                                        &U_loop[(size_t)m * (size_t)N_GRID]);

                        for (int q = 0; q < NPHI; ++q) {
                            long double Y = (long double)Z_loop[m][q] - Z0_phi[q];
                            L1_phi[m][q] += Y;
                            L2_phi[m][q] += Y*Y;
                            L3_phi[m][q] += Y*Y*Y;
                            L4_phi[m][q] += Y*Y*Y*Y;
                        }
                    }
                }
            }
        } /* end parallel */

    
        for (int m = 0; m < NCHK; ++m) {
            for (int q = 0; q < NPHI; ++q) {
                long double z = (long double)Z_loop[m][q];
                fprintf(fp_sampZ[m][q], "%.17e\n", (double)z);

                SZ1_phi[m][q] += z;
                SZ2_phi[m][q] += z*z;
                SZ3_phi[m][q] += z*z*z;
                SZ4_phi[m][q] += z*z*z*z;
            }
        }

        /* ==== density raw samples binary save ==== */
        if (STORE_U_SAMPLES) {
            fwrite(U_loop, sizeof(double), (size_t)NCHK * (size_t)N_GRID, fp_sampU);
        }

        /* ==== density moments / representative ==== */
        for (int m = 0; m < NCHK; ++m) {
            size_t base = (size_t)m * (size_t)N_GRID;
            for (int i = 0; i < N_GRID; ++i) {
                long double u = (long double)U_loop[base + (size_t)i];
                U1[base + (size_t)i] += u;
                U2[base + (size_t)i] += u*u;
                U3[base + (size_t)i] += u*u*u;
                U4[base + (size_t)i] += u*u*u*u;
                if (loop == 0) Urep[base + (size_t)i] = U_loop[base + (size_t)i];
            }
        }

        /* === average for mu (direct observed) === */
        long double sum_phi = 0.0L;
        #pragma omp parallel for reduction(+:sum_phi)
        for (size_t p = 0; p < Np; ++p)
            sum_phi += cos(X[p]);
        long double r = sum_phi / (long double)Np - phi_u0;
        Sr1 += r; Sr2 += r*r; Sr3 += r*r*r; Sr4 += r*r*r*r;

        /* ---- RAW: histogram ---- */
        memset(cnt, 0, N_GRID * sizeof *cnt);
        #pragma omp parallel
        {
            double *local = calloc(N_GRID, sizeof *local);
            #pragma omp for nowait
            for (size_t p = 0; p < Np; ++p) {
                int k = (int)floor(X[p] / dx);
                if (k >= N_GRID) k -= N_GRID;
                if (k < 0)       k += N_GRID;
                local[k] += 1.0;
            }
            #pragma omp critical
            {
                for (int i = 0; i < N_GRID; ++i) cnt[i] += local[i];
            }
            free(local);
        }

        /* === moment*/
        for (int i = 0; i < N_GRID; ++i) {
            long double u_raw = (long double)cnt[i] / ((long double)Np * dx);
            R1[i] += u_raw;
            R2[i] += u_raw * u_raw;
            R3[i] += u_raw * u_raw * u_raw;
            R4[i] += u_raw * u_raw * u_raw * u_raw;
        }

        for (int m = 0; m < NCHK; ++m) {
            for (int q = 0; q < NPHI; ++q) {
                S1_phi[m][q] += L1_phi[m][q];
                S2_phi[m][q] += L2_phi[m][q];
                S3_phi[m][q] += L3_phi[m][q];
                S4_phi[m][q] += L4_phi[m][q];
            }
        }

        free(U_loop);
    }
    fprintf(stderr, "\n");


    for (int m = 0; m < NCHK; ++m) {
        for (int q = 0; q < NPHI; ++q) {
            fclose(fp_sampZ[m][q]);
        }
    }
    if (fp_sampU) fclose(fp_sampU);
    if (STORE_U_SAMPLES) printf("Wrote samples_u_particles.bin\n");

    /*---------------- ρ statistics --------------*/
    long double nL = (long double)Nloop;
    long double mean_r_L = Sr1 / nL;
    long double var_r_L  = (Sr2 - Sr1 * mean_r_L) / (nL - 1.0L);
    long double mu3_r_L  = Sr3 / nL - 3 * mean_r_L * (Sr2 / nL) + 2 * pow(mean_r_L, 3);
    long double mu4_r_L  = Sr4 / nL - 4 * mean_r_L * (Sr3 / nL)
                           + 6 * pow(mean_r_L, 2) * (Sr2 / nL)
                           - 3 * pow(mean_r_L, 4);

    printf("# φ(x) = cos | α = %.3f | ρ stats after %d steps:\n", ALPHA, Nsteps);
    printf("#   mean               var                m3                 m4\n");
    printf("  %.17e  %.17e  %.17e  %.17e\n\n",
           (double)mean_r_L, (double)var_r_L,
           (double)mu3_r_L,  (double)mu4_r_L);

    /*---------------- moment for <Δφ,u>  -----------*/
    for (int q = 0; q < NPHI; ++q) {
        char fname[512];
        snprintf(fname, sizeof(fname),
            "phi_%s_moments_delta_particles_dt%s_steps%d_loop%ld_Np%zu.txt",
            phi_tag[q], dtstr, Nsteps, Nloop, Np);
        FILE *fp = fopen(fname, "w");
        if (!fp) { perror("fopen"); return 1; }
        fprintf(fp, "# t_step   t_time   mean(Δ)            var(Δ)             c3(Δ)               c4(Δ)   (phi=%s)\n",
                phi_tag[q]);

        printf("== Moments via Δ<%s,u> [particles] ==\n", phi_tag[q]);
        printf("step  time      mean(Δ)            var(Δ)             c3(Δ)               c4(Δ)\n");

        for (int m = 0; m < NCHK; ++m) {
            long double n = nL;
            long double mu = S1_phi[m][q]/n;
            long double m2 = S2_phi[m][q]/n;
            long double m3 = S3_phi[m][q]/n;
            long double m4 = S4_phi[m][q]/n;
            long double v2 = m2 - mu*mu;
            long double c3 = m3 - 3.0L*mu*m2 + 2.0L*mu*mu*mu;
            long double c4 = m4 - 4.0L*mu*m3 + 6.0L*mu*mu*m2 - 3.0L*mu*mu*mu*mu;

            int step = tchk[m];
            double timeT = step * dt;

            fprintf(fp,"%6d  %.8e  %.17e  %.17e  %.17e  %.17e\n",
                    step, timeT, (double)mu, (double)v2, (double)c3, (double)c4);
            printf("%4d  %.5g  %.17e  %.17e  %.17e  %.17e\n",
                   step, timeT, (double)mu, (double)v2, (double)c3, (double)c4);
        }
        fclose(fp);
        printf("Wrote %s\n\n", fname);
        fflush(stdout);
    }

    /*---------------- moment for Z=<φ,μ_t> ＋SE  -----------*/
    for (int q = 0; q < NPHI; ++q) {
        char fnameZ[512];
        snprintf(fnameZ, sizeof(fnameZ),
            "phi_%s_moments_Z_particles_dt%s_steps%d_loop%ld_Np%zu_alpha%s.txt",
            phi_tag[q], dtstr, Nsteps, Nloop, Np, astr);
        FILE *fpZ = fopen(fnameZ, "w");
        if (!fpZ) { perror("fopen moments_Z"); return 1; }
        fprintf(fpZ,
                "# Z=<%s, mu_t> moments across loops (particles average)\n"
                "# columns:\n"
                "# step  time  mean(Z)  var(Z)  c3(Z)  c4(Z)  se_mean(Z)  se_var(Z)[approx]\n"
                "# se_mean = sqrt(var/Nloop)\n"
                "# se_var  ≈ sqrt((mu4_central - var^2)/Nloop), using c4 as mu4_central\n",
                phi_tag[q]);

        printf("== Moments of Z=<%s,mu_t> [particles] (with SE for mean/var) ==\n", phi_tag[q]);
        printf("step  time      mean(Z)            var(Z)             c3(Z)               c4(Z)               se_mean            se_var(approx)\n");

        for (int m = 0; m < NCHK; ++m) {
            long double n = nL;
            long double mu = SZ1_phi[m][q]/n;
            long double m2 = SZ2_phi[m][q]/n;
            long double m3 = SZ3_phi[m][q]/n;
            long double m4 = SZ4_phi[m][q]/n;

            long double v2 = m2 - mu*mu;
            long double c3 = m3 - 3.0L*mu*m2 + 2.0L*mu*mu*mu;
            long double c4 = m4 - 4.0L*mu*m3 + 6.0L*mu*mu*m2 - 3.0L*mu*mu*mu*mu;

            long double se_mean = (v2 > 0.0L) ? sqrtl(v2 / n) : NAN;

            long double se_var = NAN;
            long double tmp = c4 - v2*v2;
            if (tmp > 0.0L) se_var = sqrtl(tmp / n);

            int step = tchk[m];
            double timeT = step * dt;

            fprintf(fpZ,"%6d  %.8e  %.17e  %.17e  %.17e  %.17e  %.17e  %.17e\n",
                    step, timeT,
                    (double)mu, (double)v2, (double)c3, (double)c4,
                    (double)se_mean, (double)se_var);

            printf("%4d  %.5g  %.17e  %.17e  %.17e  %.17e  %.17e  %.17e\n",
                   step, timeT,
                   (double)mu, (double)v2, (double)c3, (double)c4,
                   (double)se_mean, (double)se_var);
        }
        fclose(fpZ);
        printf("Wrote %s\n\n", fnameZ);
        fflush(stdout);
    }

    /*---------------- density moments / representative / mean -----------*/
    for (int m = 0; m < NCHK; ++m) {
        int step = tchk[m];
        double timeT = step * dt;

        char fname_mom[512];
        char fname_rep[512];
        char fname_mean[512];

        snprintf(fname_mom, sizeof(fname_mom),
                 "density_moments_step%04d_particles_dt%s_steps%d_loop%ld_Np%zu_alpha%s.txt",
                 step, dtstr, Nsteps, Nloop, Np, astr);

        snprintf(fname_rep, sizeof(fname_rep),
                 "density_representative_step%04d_particles_dt%s_steps%d_loop%ld_Np%zu_alpha%s.txt",
                 step, dtstr, Nsteps, Nloop, Np, astr);

        snprintf(fname_mean, sizeof(fname_mean),
                 "density_mean_step%04d_particles_dt%s_steps%d_loop%ld_Np%zu_alpha%s.txt",
                 step, dtstr, Nsteps, Nloop, Np, astr);

        /* moments */
        {
            FILE *fp = fopen(fname_mom, "w");
            if (!fp) { perror("fopen density moments"); return 1; }
            fprintf(fp, "# density moments at step=%d time=%g\n", step, timeT);
            fprintf(fp, "# x_center mean(u) var(u) c3(u) c4(u) sem(u)\n");

            for (int i = 0; i < N_GRID; ++i) {
                size_t idx = (size_t)m * (size_t)N_GRID + (size_t)i;
                long double mu = U1[idx] / nL;
                long double m2 = U2[idx] / nL;
                long double m3 = U3[idx] / nL;
                long double m4 = U4[idx] / nL;
                long double v2 = m2 - mu*mu;
                if (v2 < 0.0L) v2 = 0.0L;
                long double c3 = m3 - 3.0L*mu*m2 + 2.0L*mu*mu*mu;
                long double c4 = m4 - 4.0L*mu*m3 + 6.0L*mu*mu*m2 - 3.0L*mu*mu*mu*mu;
                long double sem = sqrtl(v2 / nL);
                double x_center = (i + 0.5) * dx;

                fprintf(fp, "%.17e %.17e %.17e %.17e %.17e %.17e\n",
                        x_center, (double)mu, (double)v2, (double)c3, (double)c4, (double)sem);
            }
            fclose(fp);
            printf("Wrote %s\n", fname_mom);
        }

        /* representative = loop 0 */
        {
            FILE *fp = fopen(fname_rep, "w");
            if (!fp) { perror("fopen density representative"); return 1; }
            fprintf(fp, "# representative density at step=%d time=%g\n", step, timeT);
            fprintf(fp, "# representative realization = loop 0\n");
            fprintf(fp, "# x_center u_rep\n");
            for (int i = 0; i < N_GRID; ++i) {
                size_t idx = (size_t)m * (size_t)N_GRID + (size_t)i;
                double x_center = (i + 0.5) * dx;
                fprintf(fp, "%.17e %.17e\n", x_center, Urep[idx]);
            }
            fclose(fp);
            printf("Wrote %s\n", fname_rep);
        }

        /* mean density */
        {
            FILE *fp = fopen(fname_mean, "w");
            if (!fp) { perror("fopen density mean"); return 1; }
            fprintf(fp, "# mean density at step=%d time=%g\n", step, timeT);
            fprintf(fp, "# averaged over loops\n");
            fprintf(fp, "# x_center mean_u\n");
            for (int i = 0; i < N_GRID; ++i) {
                size_t idx = (size_t)m * (size_t)N_GRID + (size_t)i;
                double x_center = (i + 0.5) * dx;
                long double mu = U1[idx] / nL;
                fprintf(fp, "%.17e %.17e\n", x_center, (double)mu);
            }
            fclose(fp);
            printf("Wrote %s\n", fname_mean);
        }
    }

    /*---------------- RAW profile -------------*/
    char fname_raw[256];
    snprintf(fname_raw, sizeof(fname_raw),
             "profile_raw_Np%ld_loop%ld_dt%.0e_steps%d.txt",
             Np, Nloop, dt, Nsteps);
    FILE *fp_raw = fopen(fname_raw, "w");
    if (!fp_raw) { perror("fopen"); return 1; }
    fprintf(fp_raw, "# x_center     mean_RAW             var_RAW              m3_RAW               m4_RAW\n");
    for (int i = 0; i < N_GRID; ++i) {
        long double mu  = R1[i] / nL;
        long double v2  = R2[i] / nL - mu * mu;
        long double v3  = R3[i] / nL - 3 * mu * (R2[i] / nL) + 2 * mu * mu * mu;
        long double v4  = R4[i] / nL - 4 * mu * (R3[i] / nL)
                          + 6 * mu * mu * (R2[i] / nL)
                          - 3 * mu * mu * mu * mu;
        double x_center = (i + 0.5) * dx;
        fprintf(fp_raw, "%.8e  %.17e  %.17e  %.17e  %.17e\n",
                x_center, (double)mu, (double)v2, (double)v3, (double)v4);
    }
    fclose(fp_raw);
    printf("wrote %s (RAW only)\n", fname_raw);

    /*---------------- cleanup -------------------------------*/
    free(x_grid); free(u0); free(cdf); free(t2m);
    free(X0); free(X);
    free(R1); free(R2); free(R3); free(R4);
    free(cnt);
    free(U1); free(U2); free(U3); free(U4);
    free(Urep);
    return 0;
}

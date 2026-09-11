/*  ------------------------------------------------------------------
    stochastic_fractional_diffusion_omp_observe_moments_Z_save_samples.c

    
   ------------------------------------------------------------------ */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <string.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <omp.h>
#include <gsl/gsl_errno.h>

#define N  ((1<<8) + 1)   /* 257 */

/* ---- parameter ---- */
const double L       = 2.0 * M_PI;
const double dx      = L / N;
const double dt      = 1.0e-4;
const int    STEPS   = 1;
const int    LOOP    = 1000;
const double D1      = 0.5;
const int    MTRIALS = 1000;
const double JITTER  = 1e-12;

/* Initial fluctuation flag  */
const int INIT_WITH_NOISE = 0;

/* ---- Sample storage ---- */
#define STORE_Z_SAMPLES 1
#define STORE_U_SAMPLES 1

/* ---- Initial condition options ---- */
const int    USE_HISTOGRAM_INITIAL_CONDITION = 0;
/* Number of particles used to create the initial histogram in each loop */
const size_t HIST_INIT_NP = MTRIALS;


#ifdef FFTW_NO_Complex
  #define C_RE(z)        ((z)[0])
  #define C_IM(z)        ((z)[1])
  #define C_SET(z,r,i)   do{ (z)[0]=(r); (z)[1]=(i); }while(0)
#else
  #define C_RE(z)        (creal(z))
  #define C_IM(z)        (cimag(z))
  #define C_SET(z,r,i)   do{ (z) = (r) + I*(i); }while(0)
#endif
/* ============================================================ */


int karr[N];
static inline void build_k(void){
    for(int i=0;i<N;++i){
        double f = (i <= N/2) ? (double)i/(N*dx)
                              : (double)(i-N)/(N*dx);
        karr[i] = (int)round(2.0*M_PI*f);
    }
}


static inline int find_index(int *arr,int n,int key){
    for(int i=0;i<n;++i) if(arr[i]==key) return i;
    return -1;
}

/* ---- Construct the normalized analytic initial profile u(x)=exp(-(x-L/2)^2)+1 */
static void build_analytic_initial_profile(const double *x, double *u_out)
{
    double s = 0.0;
    for(int i=0;i<N;++i){
        double d = x[i] - L/2.0;
        u_out[i] = exp(-d*d) + 1.0;
        s += u_out[i] * dx;
    }
    for(int i=0;i<N;++i) u_out[i] /= s;
}

/* Build the CDF of the analytic initial profile */
static void build_cdf_from_density(const double *u_ref, double *cdf)
{
    cdf[0] = 0.0;
    for(int i=0;i<N;++i) cdf[i+1] = cdf[i] + u_ref[i] * dx;
    cdf[N] = 1.0;
}

/* ---- Create u0 by inverse-transform sampling from the CDF + NGP histogram ---- */
static void build_histogram_initial_profile_from_cdf(const gsl_rng *rng,
                                                     const double *x,
                                                     const double *u_ref,
                                                     const double *cdf,
                                                     double *u_hist,
                                                     size_t np)
{
    double *cnt = (double*)calloc(N, sizeof(double));
    if(!cnt){
        fprintf(stderr, "calloc failed in build_histogram_initial_profile_from_cdf\n");
        exit(EXIT_FAILURE);
    }

    for(size_t p=0; p<np; ++p){
        double U = gsl_rng_uniform(rng); /* [0,1) */

        int lo = 0, hi = N;
        while(hi - lo > 1){
            int m = (lo + hi) >> 1;
            if(cdf[m] < U) lo = m;
            else hi = m;
        }

        double xpos;
        if (u_ref[lo] > 0.0) xpos = x[lo] + (U - cdf[lo]) / u_ref[lo];
        else                 xpos = x[lo];

        while(xpos >= L) xpos -= L;
        while(xpos <  0.0) xpos += L;

        int k = (int)floor(xpos / dx);
        if(k >= N) k -= N;
        if(k <  0) k += N;
        cnt[k] += 1.0;
    }

    for(int i=0;i<N;++i){
        u_hist[i] = cnt[i] / ((double)np * dx);
    }

    free(cnt);
}


static void write_profile_xy(const char *fname, const double *x, const double *u)
{
    FILE *fp = fopen(fname, "w");
    if(!fp){
        fprintf(stderr, "failed to open %s\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "# x u\n");
    for(int i=0;i<N;++i){
        fprintf(fp, "%.17e %.17e\n", x[i], u[i]);
    }
    fclose(fp);
    printf("Wrote %s\n", fname);
}

/* ---- L × g,（g~N(0,1)） ---- */
static void sample_gaussian_chol(const gsl_rng *r,
                                 const gsl_matrix *L_,
                                 gsl_vector *out)
{
    size_t dim = L_->size1;

    static _Thread_local gsl_vector *g = NULL;
    if (!g || g->size != dim){
        if (g) gsl_vector_free(g);
        g = gsl_vector_alloc(dim);
    }
    for (size_t j = 0; j < dim; ++j)
        gsl_vector_set(g, j, gsl_ran_gaussian(r, 1.0));

    for (size_t i = 0; i < dim; ++i){
        double sum = 0.0;
        for (size_t j = 0; j <= i; ++j)
            sum += gsl_matrix_get(L_, i, j) * gsl_vector_get(g, j);
        gsl_vector_set(out, i, sum);
    }
}

/* ---- Measure the covariance-matrix asymmetry max_{i<j}|C_ij-C_ji| ---- */
static double matrix_max_asymmetry(const gsl_matrix *A)
{
    size_t dim = A->size1;
    double max_asym = 0.0;
    for (size_t i = 0; i < dim; ++i){
        for (size_t j = i + 1; j < dim; ++j){
            double aij = gsl_matrix_get(A, i, j);
            double aji = gsl_matrix_get(A, j, i);
            double d = fabs(aij - aji);
            if (d > max_asym) max_asym = d;
        }
    }
    return max_asym;
}

static void inspect_density_and_save(const char *tag,
                                     int loop_id,
                                     int step,
                                     const double *x,
                                     fftw_complex *u_hat_src,
                                     fftw_complex *u_hat_work,
                                     double *u_real,
                                     fftw_plan pb)
{
    memcpy(u_hat_work, u_hat_src, N * sizeof(*u_hat_work));
    fftw_execute(pb);

    double min_rho = 1.0e300;
    double max_rho = -1.0e300;
    double neg_mass = 0.0;
    double neg_l2_sq = 0.0;

    for(int i=0;i<N;++i){
        u_real[i] /= (N*dx);
        if (u_real[i] < min_rho) min_rho = u_real[i];
        if (u_real[i] > max_rho) max_rho = u_real[i];
        if (u_real[i] < 0.0){
            double m = -u_real[i];
            neg_mass += m * dx;
            neg_l2_sq += m * m * dx;
        }
    }

    fprintf(stderr,
            "[%s] loop=%d step=%d min_rho=%.16e max_rho=%.16e neg_mass=%.16e neg_l2=%.16e\n",
            tag, loop_id, step, min_rho, max_rho, neg_mass, sqrt(neg_l2_sq));

    char fname[256];
    snprintf(fname, sizeof(fname),
             "density_snapshot_%s_loop%d_step%d.txt",
             tag, loop_id, step);

    FILE *fp = fopen(fname, "w");
    if (fp){
        fprintf(fp, "# x rho\n");
        for(int i=0;i<N;++i){
            fprintf(fp, "%.16e %.16e\n", x[i], u_real[i]);
        }
        fclose(fp);
        fprintf(stderr, "Wrote %s\n", fname);
    } else {
        fprintf(stderr, "Failed to write %s\n", fname);
    }
}

/* Build the covariance matrix C(u) */
static void build_cov(fftw_complex *u_src,double alpha,gsl_matrix *cov)
{
    int Nk = N-1;

    fftw_complex *pp = calloc(N*N,sizeof(*pp));
    fftw_complex *pm = calloc(N*N,sizeof(*pm));
    fftw_complex *mp = calloc(N*N,sizeof(*mp));
    fftw_complex *mm = calloc(N*N,sizeof(*mm));

    for(int i=0;i<N;i++){
        int s1 = karr[i];
        for(int j=0;j<N;j++){
            int s2 = karr[j];
            int ipp=find_index(karr,N,s1+s2);
            int ipm=find_index(karr,N,s1-s2);
            int imp=find_index(karr,N,-s1+s2);
            int imm=find_index(karr,N,-s1-s2);
            pp[i*N+j] = (ipp>=0 ? u_src[ipp] : 0.0+0.0*I);
            pm[i*N+j] = (ipm>=0 ? u_src[ipm] : 0.0+0.0*I);
            mp[i*N+j] = (imp>=0 ? u_src[imp] : 0.0+0.0*I);
            mm[i*N+j] = (imm>=0 ? u_src[imm] : 0.0+0.0*I);
        }
    }

    double *RR=malloc(N*N*sizeof(double));
    double *RI=malloc(N*N*sizeof(double));
    double *IR=malloc(N*N*sizeof(double));
    double *II=malloc(N*N*sizeof(double));

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            double k1=fabs((double)karr[i]);
            double k2=fabs((double)karr[j]);
            double kp=fabs((double)(karr[i]+karr[j]));
            double km=fabs((double)(karr[i]-karr[j]));
            RR[i*N+j]=-pow(kp,alpha)+pow(k1,alpha)+pow(k2,alpha);
            RI[i*N+j]=-pow(km,alpha)+pow(k1,alpha)+pow(k2,alpha);
            IR[i*N+j]=RI[i*N+j];
            II[i*N+j]=-pow(kp,alpha)+pow(k1,alpha)+pow(k2,alpha);
        }
    }

    fftw_complex *C11=calloc(N*N,sizeof(*C11));
    fftw_complex *C12=calloc(N*N,sizeof(*C12));
    fftw_complex *C21=calloc(N*N,sizeof(*C21));
    fftw_complex *C22=calloc(N*N,sizeof(*C22));

    for(int idx=0; idx<N*N; idx++){
        fftw_complex a=pp[idx], b=pm[idx], c=mp[idx], d=mm[idx];
        double r = RR[idx], ri = RI[idx], ir = IR[idx], ii = II[idx];
        C11[idx]=0.25*( a*r + d*ii + b*ri + c*ir);
        C22[idx]=0.25*(-a*r - d*ii + b*ri + c*ir);
        C12[idx]=-0.25*I*( a*r - d*ii - b*ri + c*ir);
        C21[idx]=-0.25*I*( a*r - d*ii + b*ri - c*ir);
    }

    for(int i=1;i<N;i++){
        for(int j=1;j<N;j++){
            int ii=i-1, jj=j-1;
            double A=creal(C11[i*N+j])*0.5;
            double B=creal(C12[i*N+j])*0.5;
            double C=creal(C21[i*N+j])*0.5;
            double D=creal(C22[i*N+j])*0.5;
            if(ii==jj)           A+=JITTER;
            if(ii==(jj+Nk))      B+=JITTER;
            if((ii+Nk)==jj)      C+=JITTER;
            if((ii+Nk)==(jj+Nk)) D+=JITTER;

            gsl_matrix_set(cov,   ii,    jj, A);
            gsl_matrix_set(cov,   ii,  Nk+jj, B);
            gsl_matrix_set(cov, Nk+ii,    jj, C);
            gsl_matrix_set(cov, Nk+ii,  Nk+jj, D);
        }
    }

    free(pp);free(pm);free(mp);free(mm);
    free(RR);free(RI);free(IR);free(II);
    free(C11);free(C12);free(C21);free(C22);
}

/* ---- Add initial fluctuations ---- */
static void add_initial_noise(const gsl_rng *rng,
                       fftw_complex *u_hat_base,
                       fftw_complex *u_hat_tgt,
                       double alpha,
                       gsl_matrix *cov,
                       gsl_matrix *chol,
                       gsl_vector *rv,
                       double scale)
{
    int Nk = N-1;
    gsl_matrix_set_zero(cov);
    build_cov(u_hat_base,alpha,cov);
    gsl_matrix_memcpy(chol,cov);
    if (gsl_linalg_cholesky_decomp(chol)) {
        double max_asym = matrix_max_asymmetry(cov);
        fprintf(stderr,"Cholesky failed in add_initial_noise\n");
        fprintf(stderr,"max asymmetry = %.16e\n", max_asym);
        exit(EXIT_FAILURE);
    }
    sample_gaussian_chol(rng,chol,rv);

    fftw_complex z[N]; z[0]=0.0+0.0*I;
    for(int i=1;i<N;++i){
        double re=gsl_vector_get(rv,i-1);
        double im=gsl_vector_get(rv,Nk+i-1);
        z[i]=re+I*im;
    }
    for(int i=0;i<N;++i)
        u_hat_tgt[i] += scale * z[i];
}

/* ====================================================================== */
int main(void)
{
    /* ---------- FFTW thread initialization ---------- */
    gsl_set_error_handler_off();
    fftw_init_threads();

    /* ---- Coordinates and the "legacy" φ=cos(x) (for rho_*: unchanged) ---- */
    double *x   = (double*)malloc(N*sizeof(double));
    double *phi = (double*)malloc(N*sizeof(double));
    for(int i=0;i<N;++i){ x[i]=i*dx; phi[i]=cos(x[i]); }

    build_k();

    /* ---- Analytic initial condition and its CDF (reference for regenerating the histogram in each loop) ---- */
    double *u_init_analytic   = (double*)malloc(N*sizeof(double));
    double *cdf_init_analytic = (double*)malloc((N+1)*sizeof(double));
    if(!u_init_analytic || !cdf_init_analytic){
        fprintf(stderr, "malloc failed for analytic initial profile / cdf\n");
        exit(EXIT_FAILURE);
    }

    build_analytic_initial_profile(x, u_init_analytic);
    build_cdf_from_density(u_init_analytic, cdf_init_analytic);
    write_profile_xy("initial_profile_analytic.txt", x, u_init_analytic);

    /* Create and save a histogram initial condition once as a representative example */
    if (USE_HISTOGRAM_INITIAL_CONDITION){
        double *u_hist_example = (double*)malloc(N*sizeof(double));
        gsl_rng *rng_example = gsl_rng_alloc(gsl_rng_default);
        gsl_rng_set(rng_example, (unsigned long)time(NULL) ^ 0x9e3779b9UL);
        build_histogram_initial_profile_from_cdf(rng_example, x,
                                                 u_init_analytic,
                                                 cdf_init_analytic,
                                                 u_hist_example,
                                                 HIST_INIT_NP);
        write_profile_xy("initial_profile_histogram_example.txt", x, u_hist_example);
        gsl_rng_free(rng_example);
        free(u_hist_example);
    }

    /* ---- rho_*（φ=cos） ---- */
    double *rho_det   = (double*)malloc(LOOP*sizeof(double));
    double *rho_self  = (double*)malloc(LOOP*sizeof(double));

    /* ----Moment output for the raw-grid profile ---- */
    double *mean_det  = (double*)calloc(N,sizeof(double));
    double *mean_self = (double*)calloc(N,sizeof(double));
    double *s1_det  = (double*)calloc(N,sizeof(double)), *s2_det  = (double*)calloc(N,sizeof(double));
    double *s3_det  = (double*)calloc(N,sizeof(double)), *s4_det  = (double*)calloc(N,sizeof(double));
    double *s1_self = (double*)calloc(N,sizeof(double)), *s2_self = (double*)calloc(N,sizeof(double));
    double *s3_self = (double*)calloc(N,sizeof(double)), *s4_self = (double*)calloc(N,sizeof(double));

    /* ---- Set of test functions ---- */
    enum { NPHI = 23, MAXCHK = 16 };
    const char *phi_tag[NPHI] = {
        "cosx","cos2x","cos4x","cos6x","cos8x","cos10x","cos12x","cos14x","cos16x","cos64x","cos256x",
        "sinx","sin2x","sin4x","sin6x","sin8x","sin10x","sin12x","sin14x","sin16x","sin64x","sin256x",
        "gauss_center"
    };

    /* Precompute φ_k(x_j) */
    double *phi_set[NPHI];
    for(int p=0;p<NPHI;++p) phi_set[p] = (double*)malloc(N*sizeof(double));
    for(int j=0;j<N;++j){
        double xx = x[j];
        double dcenter = xx - L/2.0;

        phi_set[0][j]  = cos(1.0*xx);
        phi_set[1][j]  = cos(2.0*xx);
        phi_set[2][j]  = cos(4.0*xx);
        phi_set[3][j]  = cos(6.0*xx);
        phi_set[4][j]  = cos(8.0*xx);
        phi_set[5][j]  = cos(10.0*xx);
        phi_set[6][j]  = cos(12.0*xx);
        phi_set[7][j]  = cos(14.0*xx);
        phi_set[8][j]  = cos(16.0*xx);
        phi_set[9][j]  = cos(64.0*xx);
        phi_set[10][j] = cos(256.0*xx);

        phi_set[11][j] = sin(1.0*xx);
        phi_set[12][j] = sin(2.0*xx);
        phi_set[13][j] = sin(4.0*xx);
        phi_set[14][j] = sin(6.0*xx);
        phi_set[15][j] = sin(8.0*xx);
        phi_set[16][j] = sin(10.0*xx);
        phi_set[17][j] = sin(12.0*xx);
        phi_set[18][j] = sin(14.0*xx);
        phi_set[19][j] = sin(16.0*xx);
        phi_set[20][j] = sin(64.0*xx);
        phi_set[21][j] = sin(256.0*xx);

        phi_set[22][j] = exp(-dcenter*dcenter);
    }

    int tchk[MAXCHK];
    int NCHK = 0;
    const int base[] = {0,1,20,50,100,200};
    const int base_len = (int)(sizeof(base)/sizeof(base[0]));
    for(int i=0;i<base_len && NCHK<MAXCHK; ++i){
        if (base[i] <= STEPS) tchk[NCHK++] = base[i];
    }
    int found = 0; for(int i=0;i<NCHK; ++i) if (tchk[i]==STEPS) { found=1; break; }
    if (!found && NCHK<MAXCHK) tchk[NCHK++] = STEPS;

    fprintf(stdout, "# Checkpoints (%d):", NCHK);
    for(int i=0;i<NCHK;++i) fprintf(stdout, " %d(%.4g)", tchk[i], tchk[i]*dt);
    fprintf(stdout, "\n"); fflush(stdout);

    /* Raw-moment accumulation for <Zφ,u> (det/self × time × φ) */
    long double S1_det[MAXCHK][NPHI]; long double S2_det[MAXCHK][NPHI];
    long double S3_det[MAXCHK][NPHI]; long double S4_det[MAXCHK][NPHI];
    long double S1_self[MAXCHK][NPHI]; long double S2_self[MAXCHK][NPHI];
    long double S3_self[MAXCHK][NPHI]; long double S4_self[MAXCHK][NPHI];
    memset(S1_det,0,sizeof(S1_det)); memset(S2_det,0,sizeof(S2_det));
    memset(S3_det,0,sizeof(S3_det)); memset(S4_det,0,sizeof(S4_det));
    memset(S1_self,0,sizeof(S1_self)); memset(S2_self,0,sizeof(S2_self));
    memset(S3_self,0,sizeof(S3_self)); memset(S4_self,0,sizeof(S4_self));

    /* Raw-moment accumulation for u(x,t) (det/self × time × grid point) */
    long double U1_det[MAXCHK][N]; long double U2_det[MAXCHK][N];
    long double U3_det[MAXCHK][N]; long double U4_det[MAXCHK][N];
    long double U1_self[MAXCHK][N]; long double U2_self[MAXCHK][N];
    long double U3_self[MAXCHK][N]; long double U4_self[MAXCHK][N];
    memset(U1_det,0,sizeof(U1_det)); memset(U2_det,0,sizeof(U2_det));
    memset(U3_det,0,sizeof(U3_det)); memset(U4_det,0,sizeof(U4_det));
    memset(U1_self,0,sizeof(U1_self)); memset(U2_self,0,sizeof(U2_self));
    memset(U3_self,0,sizeof(U3_self)); memset(U4_self,0,sizeof(U4_self));

    /* ----Representative single density at each checkpoint ---- */
    double *Urep_det  = (double*)malloc((size_t)NCHK * (size_t)N * sizeof(double));
    double *Urep_self = (double*)malloc((size_t)NCHK * (size_t)N * sizeof(double));
    if(!Urep_det || !Urep_self){
        fprintf(stderr, "malloc failed for Urep_det / Urep_self\n");
        exit(EXIT_FAILURE);
    }

#if STORE_Z_SAMPLES
    size_t Zsz = (size_t)LOOP * (size_t)NCHK * (size_t)NPHI;
    double *Z_det_samples  = (double*)malloc(Zsz * sizeof(double));
    double *Z_self_samples = (double*)malloc(Zsz * sizeof(double));
    if(!Z_det_samples || !Z_self_samples){
        fprintf(stderr,"malloc failed for Z_det_samples / Z_self_samples\n");
        exit(EXIT_FAILURE);
    }
#endif

#if STORE_U_SAMPLES
    size_t Usz = (size_t)LOOP * (size_t)NCHK * (size_t)N;
    double *U_det_samples  = (double*)malloc(Usz * sizeof(double));
    double *U_self_samples = (double*)malloc(Usz * sizeof(double));
    if(!U_det_samples || !U_self_samples){
        fprintf(stderr,"malloc failed for U_det_samples / U_self_samples\n");
        exit(EXIT_FAILURE);
    }
#endif

    int Nk = N-1, dim = 2*Nk;
    double alpha = 1.0;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();

        gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
        gsl_rng_set(rng, (unsigned long)(time(NULL)) + 10007UL*(unsigned long)(tid+1));

        /* Real-space/Fourier-space buffers */
        double *u0 = (double*)malloc(N*sizeof(double));
        fftw_complex *u_hat         = fftw_malloc(N*sizeof(*u_hat));
        fftw_complex *u_hat_det     = fftw_malloc(N*sizeof(*u_hat_det));
        fftw_complex *u_hat_detCov  = fftw_malloc(N*sizeof(*u_hat_detCov));
        fftw_complex *u_hat_selfCov = fftw_malloc(N*sizeof(*u_hat_selfCov));

        /* FFTW  */
        fftw_plan pf, pb;
#pragma omp critical(fftw_plan)
        {
            fftw_plan_with_nthreads(1);
            pf = fftw_plan_dft_r2c_1d(N,u0,u_hat,FFTW_ESTIMATE);
            pb = fftw_plan_dft_c2r_1d(N,u_hat,u0,FFTW_ESTIMATE);
        }

        gsl_matrix *cov  = gsl_matrix_alloc(dim,dim);
        gsl_matrix *chol = gsl_matrix_alloc(dim,dim);
        gsl_vector *rv   = gsl_vector_alloc(dim);

        double decay[N];

        /* Profile statistics (raw grid)*/
        double mean_det_loc [N];  memset(mean_det_loc,  0, sizeof(mean_det_loc));
        double mean_self_loc[N];  memset(mean_self_loc, 0, sizeof(mean_self_loc));
        double s1_det_loc[N];  memset(s1_det_loc,  0, sizeof(s1_det_loc));
        double s2_det_loc[N];  memset(s2_det_loc,  0, sizeof(s2_det_loc));
        double s3_det_loc[N];  memset(s3_det_loc,  0, sizeof(s3_det_loc));
        double s4_det_loc[N];  memset(s4_det_loc,  0, sizeof(s4_det_loc));
        double s1_self_loc[N]; memset(s1_self_loc, 0, sizeof(s1_self_loc));
        double s2_self_loc[N]; memset(s2_self_loc, 0, sizeof(s2_self_loc));
        double s3_self_loc[N]; memset(s3_self_loc, 0, sizeof(s3_self_loc));
        double s4_self_loc[N]; memset(s4_self_loc, 0, sizeof(s4_self_loc));

        /* Raw moments of <Zφ,u> */
        long double S1_det_loc[MAXCHK][NPHI]; long double S2_det_loc[MAXCHK][NPHI];
        long double S3_det_loc[MAXCHK][NPHI]; long double S4_det_loc[MAXCHK][NPHI];
        long double S1_self_loc[MAXCHK][NPHI]; long double S2_self_loc[MAXCHK][NPHI];
        long double S3_self_loc[MAXCHK][NPHI]; long double S4_self_loc[MAXCHK][NPHI];
        memset(S1_det_loc,0,sizeof(S1_det_loc)); memset(S2_det_loc,0,sizeof(S2_det_loc));
        memset(S3_det_loc,0,sizeof(S3_det_loc)); memset(S4_det_loc,0,sizeof(S4_det_loc));
        memset(S1_self_loc,0,sizeof(S1_self_loc)); memset(S2_self_loc,0,sizeof(S2_self_loc));
        memset(S3_self_loc,0,sizeof(S3_self_loc)); memset(S4_self_loc,0,sizeof(S4_self_loc));

        /* Raw moments of u(x,t) */
        long double U1_det_loc[MAXCHK][N]; long double U2_det_loc[MAXCHK][N];
        long double U3_det_loc[MAXCHK][N]; long double U4_det_loc[MAXCHK][N];
        long double U1_self_loc[MAXCHK][N]; long double U2_self_loc[MAXCHK][N];
        long double U3_self_loc[MAXCHK][N]; long double U4_self_loc[MAXCHK][N];
        memset(U1_det_loc,0,sizeof(U1_det_loc)); memset(U2_det_loc,0,sizeof(U2_det_loc));
        memset(U3_det_loc,0,sizeof(U3_det_loc)); memset(U4_det_loc,0,sizeof(U4_det_loc));
        memset(U1_self_loc,0,sizeof(U1_self_loc)); memset(U2_self_loc,0,sizeof(U2_self_loc));
        memset(U3_self_loc,0,sizeof(U3_self_loc)); memset(U4_self_loc,0,sizeof(U4_self_loc));

#pragma omp for schedule(static)
        for(int n=0;n<LOOP;++n){
            if (tid == 0){
                printf("loop %d / %d\r", n+1, LOOP);
                fflush(stdout);
            }

            long double phi_u0 = 0.0L;

            if (USE_HISTOGRAM_INITIAL_CONDITION){
                /* ---- Create a new initial histogram with fresh random numbers for each loop ---- */
                build_histogram_initial_profile_from_cdf(rng, x,
                                                         u_init_analytic,
                                                         cdf_init_analytic,
                                                         u0,
                                                         HIST_INIT_NP);
                for(int i=0;i<N;++i){
                    phi_u0 += (long double)phi[i] * (long double)u0[i];
                }
                phi_u0 *= dx;
            } else {
                /* ---- Original analytic initial condition ---- */
                for(int i=0;i<N;++i){
                    double d=x[i]-L/2.0;
                    u0[i]=exp(-d*d)+1.0;
                }
                double s_norm=0.0;
                for(int i=0;i<N;++i){
                    s_norm += u0[i]*dx;
                    phi_u0 += (long double)phi[i]*(long double)u0[i]*(long double)dx;
                }
                for(int i=0;i<N;++i) u0[i]/=s_norm;
            }

            /* FFT */
            fftw_execute(pf);
            for(int i=0;i<=N/2;++i){
                fftw_complex c=u_hat[i]*dx;
                u_hat_det[i]=c;
                u_hat_detCov[i]=c;
                u_hat_selfCov[i]=c;
            }
            for(int i=1;i<=N/2;++i){
                u_hat_det[N-i]=conj(u_hat_det[i]);
                u_hat_detCov[N-i]=conj(u_hat_detCov[i]);
                u_hat_selfCov[N-i]=conj(u_hat_selfCov[i]);
            }

            if (INIT_WITH_NOISE){
                double scale0 = 1.0/sqrt((double)MTRIALS);
                add_initial_noise(rng,u_hat_det,u_hat_detCov,alpha,
                                  cov,chol,rv,scale0);
                add_initial_noise(rng,u_hat_det,u_hat_selfCov,alpha,
                                  cov,chol,rv,scale0);
            }

            for(int i=0;i<N;++i)
                decay[i]=exp(-D1*pow(fabs((double)karr[i]),alpha)*dt);

         
            int chk_idx = 0;
            if (chk_idx < NCHK && tchk[chk_idx] == 0){
                /* detCov at step 0 */
                memcpy(u_hat, u_hat_detCov, N*sizeof(*u_hat));
                fftw_execute(pb);
                for(int i=0;i<N;++i) u0[i]/=(N*dx);

                if (n == 0){
                    size_t base = (size_t)chk_idx * (size_t)N;
                    for(int j=0;j<N;++j) Urep_det[base + (size_t)j] = u0[j];
                }

                for(int j=0;j<N;++j){
                    long double Y = (long double)u0[j];
                    U1_det_loc[chk_idx][j] += Y;
                    U2_det_loc[chk_idx][j] += Y*Y;
                    U3_det_loc[chk_idx][j] += Y*Y*Y;
                    U4_det_loc[chk_idx][j] += Y*Y*Y*Y;
#if STORE_U_SAMPLES
                    size_t idxu = ((size_t)n*(size_t)NCHK + (size_t)chk_idx)*(size_t)N + (size_t)j;
                    U_det_samples[idxu] = u0[j];
#endif
                }

                for(int p=0;p<NPHI;++p){
                    long double Z=0.0L;
                    for(int j=0;j<N;++j) Z += (long double)phi_set[p][j]*(long double)u0[j];
                    Z *= dx;

                    long double Y = Z;
                    S1_det_loc[chk_idx][p] += Y;
                    S2_det_loc[chk_idx][p] += Y*Y;
                    S3_det_loc[chk_idx][p] += Y*Y*Y;
                    S4_det_loc[chk_idx][p] += Y*Y*Y*Y;
#if STORE_Z_SAMPLES
                    size_t idx = ((size_t)n*(size_t)NCHK + (size_t)chk_idx)*(size_t)NPHI + (size_t)p;
                    Z_det_samples[idx] = (double)Z;
#endif
                }

                /* selfCov at step 0 */
                memcpy(u_hat, u_hat_selfCov, N*sizeof(*u_hat));
                fftw_execute(pb);
                for(int i=0;i<N;++i) u0[i]/=(N*dx);

                if (n == 0){
                    size_t base = (size_t)chk_idx * (size_t)N;
                    for(int j=0;j<N;++j) Urep_self[base + (size_t)j] = u0[j];
                }

                for(int j=0;j<N;++j){
                    long double Y = (long double)u0[j];
                    U1_self_loc[chk_idx][j] += Y;
                    U2_self_loc[chk_idx][j] += Y*Y;
                    U3_self_loc[chk_idx][j] += Y*Y*Y;
                    U4_self_loc[chk_idx][j] += Y*Y*Y*Y;
#if STORE_U_SAMPLES
                    size_t idxu = ((size_t)n*(size_t)NCHK + (size_t)chk_idx)*(size_t)N + (size_t)j;
                    U_self_samples[idxu] = u0[j];
#endif
                }

                for(int p=0;p<NPHI;++p){
                    long double Z=0.0L;
                    for(int j=0;j<N;++j) Z += (long double)phi_set[p][j]*(long double)u0[j];
                    Z *= dx;

                    long double Y = Z;
                    S1_self_loc[chk_idx][p] += Y;
                    S2_self_loc[chk_idx][p] += Y*Y;
                    S3_self_loc[chk_idx][p] += Y*Y*Y;
                    S4_self_loc[chk_idx][p] += Y*Y*Y*Y;
#if STORE_Z_SAMPLES
                    size_t idx = ((size_t)n*(size_t)NCHK + (size_t)chk_idx)*(size_t)NPHI + (size_t)p;
                    Z_self_samples[idx] = (double)Z;
#endif
                }

                ++chk_idx;
            }

            /* -------- time evolution -------- */
            for(int t=0;t<STEPS;++t){
                /* detCov の一歩 */
                gsl_matrix_set_zero(cov);
                build_cov(u_hat_det,alpha,cov);
                gsl_matrix_memcpy(chol,cov);
                if (gsl_linalg_cholesky_decomp(chol)) {
                    double max_asym = matrix_max_asymmetry(cov);
                    fprintf(stderr,
                            "Cholesky failed in detCov: loop=%d step=%d\n",
                            n, t+1);
                    fprintf(stderr, "max asymmetry = %.16e\n", max_asym);
                    exit(EXIT_FAILURE);
                }
                sample_gaussian_chol(rng,chol,rv);

                fftw_complex z[N]; z[0]=0.0+0.0*I;
                for(int i=1;i<N;++i){
                    double re=gsl_vector_get(rv,i-1);
                    double im=gsl_vector_get(rv,Nk+i-1);
                    z[i]=re+I*im;
                }
                for(int i=0;i<N;++i){
                    u_hat_det[i]*=decay[i];
                    u_hat_detCov[i]=u_hat_detCov[i]*decay[i]
                                   +sqrt(dt)/sqrt(MTRIALS)*z[i];
                }

                /* one-step for selfCov  */
                gsl_matrix_set_zero(cov);
                build_cov(u_hat_selfCov,alpha,cov);
                gsl_matrix_memcpy(chol,cov);
                if (gsl_linalg_cholesky_decomp(chol)) {
                    double max_asym = matrix_max_asymmetry(cov);
                    fprintf(stderr,
                            "Cholesky failed in selfCov: loop=%d step=%d\n",
                            n, t+1);
                    fprintf(stderr, "max asymmetry = %.16e\n", max_asym);

#pragma omp critical
                    {
                        inspect_density_and_save("self_before_break",
                                                 n, t+1,
                                                 x,
                                                 u_hat_selfCov,
                                                 u_hat,
                                                 u0,
                                                 pb);
                    }

                    exit(EXIT_FAILURE);
                }
                sample_gaussian_chol(rng,chol,rv);

                z[0]=0.0+0.0*I;
                for(int i=1;i<N;++i){
                    double re=gsl_vector_get(rv,i-1);
                    double im=gsl_vector_get(rv,Nk+i-1);
                    z[i]=re+I*im;
                }
                for(int i=0;i<N;++i)
                    u_hat_selfCov[i]=u_hat_selfCov[i]*decay[i]
                                     +sqrt(dt)/sqrt(MTRIALS)*z[i];

                /* Observation: if at a checkpoint, measure Z=<φ,u> and u(x,t) */
                int step_now = t+1;
                if (chk_idx < NCHK && step_now == tchk[chk_idx]){
                    /* detCov  */
                    memcpy(u_hat, u_hat_detCov, N*sizeof(*u_hat));
                    fftw_execute(pb);
                    for(int i=0;i<N;++i) u0[i]/=(N*dx);

                    if (n == 0){
                        size_t base = (size_t)chk_idx * (size_t)N;
                        for(int j=0;j<N;++j) Urep_det[base + (size_t)j] = u0[j];
                    }

                    for(int j=0;j<N;++j){
                        long double Y = (long double)u0[j];
                        U1_det_loc[chk_idx][j] += Y;
                        U2_det_loc[chk_idx][j] += Y*Y;
                        U3_det_loc[chk_idx][j] += Y*Y*Y;
                        U4_det_loc[chk_idx][j] += Y*Y*Y*Y;
#if STORE_U_SAMPLES
                        size_t idxu = ((size_t)n*(size_t)NCHK + (size_t)chk_idx)*(size_t)N + (size_t)j;
                        U_det_samples[idxu] = u0[j];
#endif
                    }

                    for(int p=0;p<NPHI;++p){
                        long double Z=0.0L;
                        for(int j=0;j<N;++j) Z += (long double)phi_set[p][j]*(long double)u0[j];
                        Z *= dx;

                        long double Y = Z;
                        S1_det_loc[chk_idx][p] += Y;
                        S2_det_loc[chk_idx][p] += Y*Y;
                        S3_det_loc[chk_idx][p] += Y*Y*Y;
                        S4_det_loc[chk_idx][p] += Y*Y*Y*Y;
#if STORE_Z_SAMPLES
                        size_t idx = ((size_t)n*(size_t)NCHK + (size_t)chk_idx)*(size_t)NPHI + (size_t)p;
                        Z_det_samples[idx] = (double)Z;
#endif
                    }

                    /* selfCov  */
                    memcpy(u_hat, u_hat_selfCov, N*sizeof(*u_hat));
                    fftw_execute(pb);
                    for(int i=0;i<N;++i) u0[i]/=(N*dx);

                    if (n == 0){
                        size_t base = (size_t)chk_idx * (size_t)N;
                        for(int j=0;j<N;++j) Urep_self[base + (size_t)j] = u0[j];
                    }

                    for(int j=0;j<N;++j){
                        long double Y = (long double)u0[j];
                        U1_self_loc[chk_idx][j] += Y;
                        U2_self_loc[chk_idx][j] += Y*Y;
                        U3_self_loc[chk_idx][j] += Y*Y*Y;
                        U4_self_loc[chk_idx][j] += Y*Y*Y*Y;
#if STORE_U_SAMPLES
                        size_t idxu = ((size_t)n*(size_t)NCHK + (size_t)chk_idx)*(size_t)N + (size_t)j;
                        U_self_samples[idxu] = u0[j];
#endif
                    }

                    for(int p=0;p<NPHI;++p){
                        long double Z=0.0L;
                        for(int j=0;j<N;++j) Z += (long double)phi_set[p][j]*(long double)u0[j];
                        Z *= dx;

                        long double Y = Z;
                        S1_self_loc[chk_idx][p] += Y;
                        S2_self_loc[chk_idx][p] += Y*Y;
                        S3_self_loc[chk_idx][p] += Y*Y*Y;
                        S4_self_loc[chk_idx][p] += Y*Y*Y*Y;
#if STORE_Z_SAMPLES
                        size_t idx = ((size_t)n*(size_t)NCHK + (size_t)chk_idx)*(size_t)NPHI + (size_t)p;
                        Z_self_samples[idx] = (double)Z;
#endif
                    }
                    ++chk_idx;
                }
            } /* end for t */

            /* ---- detCov in real space (final time T) → raw-grid profile statistics ---- */
            memcpy(u_hat,u_hat_detCov,N*sizeof(*u_hat));
            fftw_execute(pb);
            for(int i=0;i<N;++i) u0[i]/=(N*dx);

            long double sum_phi_u_det=0;
            for(int i=0;i<N;++i) sum_phi_u_det+=phi[i]*u0[i];
            rho_det[n]=(double)(sum_phi_u_det*dx-phi_u0);

            for(int j=0;j<N;++j){
                double v=u0[j];
                mean_det_loc[j]+=v/LOOP;
                s1_det_loc[j]+=v;  s2_det_loc[j]+=v*v;
                s3_det_loc[j]+=v*v*v; s4_det_loc[j]+=v*v*v*v;
            }

            /* selfCov in real space (final time T) → raw-grid profile statistics */
            memcpy(u_hat,u_hat_selfCov,N*sizeof(*u_hat));
            fftw_execute(pb);
            for(int i=0;i<N;++i) u0[i]/=(N*dx);

            long double sum_phi_u_self=0;
            for(int i=0;i<N;++i) sum_phi_u_self+=phi[i]*u0[i];
            rho_self[n]=(double)(sum_phi_u_self*dx-phi_u0);

            for(int j=0;j<N;++j){
                double v=u0[j];
                mean_self_loc[j]+=v/LOOP;
                s1_self_loc[j]+=v;  s2_self_loc[j]+=v*v;
                s3_self_loc[j]+=v*v*v; s4_self_loc[j]+=v*v*v*v;
            }
        } /* end for n */

        /* ---- OMP---- */
#pragma omp critical
        {
            for(int j=0;j<N;++j){
                mean_det[j]  += mean_det_loc[j];
                mean_self[j] += mean_self_loc[j];
                s1_det[j]+=s1_det_loc[j]; s2_det[j]+=s2_det_loc[j];
                s3_det[j]+=s3_det_loc[j]; s4_det[j]+=s4_det_loc[j];
                s1_self[j]+=s1_self_loc[j]; s2_self[j]+=s2_self_loc[j];
                s3_self[j]+=s3_self_loc[j]; s4_self[j]+=s4_self_loc[j];
            }

            for(int m=0;m<NCHK;++m){
                for(int p=0;p<NPHI;++p){
                    S1_det[m][p]  += S1_det_loc[m][p];
                    S2_det[m][p]  += S2_det_loc[m][p];
                    S3_det[m][p]  += S3_det_loc[m][p];
                    S4_det[m][p]  += S4_det_loc[m][p];
                    S1_self[m][p] += S1_self_loc[m][p];
                    S2_self[m][p] += S2_self_loc[m][p];
                    S3_self[m][p] += S3_self_loc[m][p];
                    S4_self[m][p] += S4_self_loc[m][p];
                }
            }

            for(int m=0;m<NCHK;++m){
                for(int j=0;j<N;++j){
                    U1_det[m][j]  += U1_det_loc[m][j];
                    U2_det[m][j]  += U2_det_loc[m][j];
                    U3_det[m][j]  += U3_det_loc[m][j];
                    U4_det[m][j]  += U4_det_loc[m][j];
                    U1_self[m][j] += U1_self_loc[m][j];
                    U2_self[m][j] += U2_self_loc[m][j];
                    U3_self[m][j] += U3_self_loc[m][j];
                    U4_self[m][j] += U4_self_loc[m][j];
                }
            }
        }


#pragma omp critical(fftw_plan)
        {
            fftw_destroy_plan(pf);
            fftw_destroy_plan(pb);
        }
        gsl_rng_free(rng);
        gsl_matrix_free(cov); gsl_matrix_free(chol);
        gsl_vector_free(rv);
        fftw_free(u_hat); fftw_free(u_hat_det);
        fftw_free(u_hat_detCov); fftw_free(u_hat_selfCov);
        free(u0);
    } /* end parallel */

    printf("\n");

    /* ----Save Z samples ---- */
#if STORE_Z_SAMPLES
    {
        FILE *fm = fopen("samples_Z_meta.txt","w");
        if(!fm){
            fprintf(stderr,"failed to open samples_Z_meta.txt\n");
            exit(EXIT_FAILURE);
        }
        fprintf(fm,"LOOP %d\n", LOOP);
        fprintf(fm,"NCHK %d\n", NCHK);
        fprintf(fm,"NPHI %d\n", NPHI);
        fprintf(fm,"dt %.17e\n", dt);
        fprintf(fm,"STEPS %d\n", STEPS);
        fprintf(fm,"alpha %.17e\n", alpha);
        fprintf(fm,"MTRIALS %d\n", MTRIALS);
        fprintf(fm,"USE_HISTOGRAM_INITIAL_CONDITION %d\n", USE_HISTOGRAM_INITIAL_CONDITION);
        fprintf(fm,"HIST_INIT_NP %zu\n", HIST_INIT_NP);
        fprintf(fm,"HISTOGRAM_REGENERATED_EACH_LOOP %d\n", USE_HISTOGRAM_INITIAL_CONDITION ? 1 : 0);
        fprintf(fm,"tchk");
        for(int m=0;m<NCHK;++m) fprintf(fm," %d", tchk[m]);
        fprintf(fm,"\n");
        fprintf(fm,"phi_tag");
        for(int p=0;p<NPHI;++p) fprintf(fm," %s", phi_tag[p]);
        fprintf(fm,"\n");
        fprintf(fm,"layout idx = (n*NCHK + m)*NPHI + p, stored as double array of length LOOP*NCHK*NPHI\n");
        fclose(fm);
        printf("Wrote samples_Z_meta.txt\n");
    }
    {
        FILE *fp = fopen("samples_Z_det.bin","wb");
        if(!fp){
            fprintf(stderr,"failed to open samples_Z_det.bin\n");
            exit(EXIT_FAILURE);
        }
        fwrite(Z_det_samples, sizeof(double), (size_t)LOOP*(size_t)NCHK*(size_t)NPHI, fp);
        fclose(fp);
        printf("Wrote samples_Z_det.bin\n");
    }
    {
        FILE *fp = fopen("samples_Z_self.bin","wb");
        if(!fp){
            fprintf(stderr,"failed to open samples_Z_self.bin\n");
            exit(EXIT_FAILURE);
        }
        fwrite(Z_self_samples, sizeof(double), (size_t)LOOP*(size_t)NCHK*(size_t)NPHI, fp);
        fclose(fp);
        printf("Wrote samples_Z_self.bin\n");
    }
#endif

    /* ---- Save u samples---- */
#if STORE_U_SAMPLES
    {
        FILE *fm = fopen("samples_u_meta.txt","w");
        if(!fm){
            fprintf(stderr,"failed to open samples_u_meta.txt\n");
            exit(EXIT_FAILURE);
        }
        fprintf(fm,"LOOP %d\n", LOOP);
        fprintf(fm,"NCHK %d\n", NCHK);
        fprintf(fm,"N %d\n", N);
        fprintf(fm,"dt %.17e\n", dt);
        fprintf(fm,"STEPS %d\n", STEPS);
        fprintf(fm,"alpha %.17e\n", alpha);
        fprintf(fm,"MTRIALS %d\n", MTRIALS);
        fprintf(fm,"dx %.17e\n", dx);
        fprintf(fm,"USE_HISTOGRAM_INITIAL_CONDITION %d\n", USE_HISTOGRAM_INITIAL_CONDITION);
        fprintf(fm,"HIST_INIT_NP %zu\n", HIST_INIT_NP);
        fprintf(fm,"HISTOGRAM_REGENERATED_EACH_LOOP %d\n", USE_HISTOGRAM_INITIAL_CONDITION ? 1 : 0);
        fprintf(fm,"REPRESENTATIVE_PROFILE loop=0 realization at each checkpoint\n");
        fprintf(fm,"MEAN_PROFILE averaged over LOOP realizations at each checkpoint\n");
        fprintf(fm,"tchk");
        for(int m=0;m<NCHK;++m) fprintf(fm," %d", tchk[m]);
        fprintf(fm,"\n");
        fprintf(fm,"layout idx = (n*NCHK + m)*N + j, stored as double array of length LOOP*NCHK*N\n");
        fclose(fm);
        printf("Wrote samples_u_meta.txt\n");
    }
    {
        FILE *fp = fopen("samples_u_det.bin","wb");
        if(!fp){
            fprintf(stderr,"failed to open samples_u_det.bin\n");
            exit(EXIT_FAILURE);
        }
        fwrite(U_det_samples, sizeof(double), (size_t)LOOP*(size_t)NCHK*(size_t)N, fp);
        fclose(fp);
        printf("Wrote samples_u_det.bin\n");
    }
    {
        FILE *fp = fopen("samples_u_self.bin","wb");
        if(!fp){
            fprintf(stderr,"failed to open samples_u_self.bin\n");
            exit(EXIT_FAILURE);
        }
        fwrite(U_self_samples, sizeof(double), (size_t)LOOP*(size_t)NCHK*(size_t)N, fp);
        fclose(fp);
        printf("Wrote samples_u_self.bin\n");
    }
#endif

    /* ----2-pass statistics for ρ ---- */
    long double mu_det_r=0.0L,mu_self_r=0.0L;
    for(int i=0;i<LOOP;++i){ mu_det_r+=rho_det[i]; mu_self_r+=rho_self[i]; }
    mu_det_r/=LOOP; mu_self_r/=LOOP;

    long double S2d=0.0L,S3d=0.0L,S4d=0.0L;
    long double S2s=0.0L,S3s=0.0L,S4s=0.0L;
    for(int i=0;i<LOOP;++i){
        long double d=rho_det[i]-mu_det_r, s=rho_self[i]-mu_self_r;
        long double d2=d*d, s2=s*s;
        S2d+=d2; S3d+=d2*d; S4d+=d2*d2;
        S2s+=s2; S3s+=s2*s; S4s+=s2*s2;
    }
    long double var_det=S2d/(LOOP-1), var_self=S2s/(LOOP-1);
    long double skew_det=(S3d/LOOP)/powl(var_det,1.5L);
    long double skew_self=(S3s/LOOP)/powl(var_self,1.5L);
    long double kurt_det=(S4d/LOOP)/(var_det*var_det);
    long double kurt_self=(S4s/LOOP)/(var_self*var_self);

    printf("det-cov : mean=%Le var=%Le skew=%Le kurt=%Le\n",
           mu_det_r,var_det,skew_det,kurt_det);
    printf("self-cov: mean=%Le var=%Le skew=%Le kurt=%Le\n",
           mu_self_r,var_self,skew_self,kurt_self);

    long double mu3_det  = skew_det  * powl(var_det , 1.5L);
    long double mu4_det  = kurt_det  * (var_det  * var_det);
    long double mu3_self = skew_self * powl(var_self, 1.5L);
    long double mu4_self = kurt_self * (var_self * var_self);

    printf("# φ(x) = cos | α = %.3f | ρ stats after %d steps (det-cov):\n",
           (double)alpha, STEPS);
    printf("#   mean               var                m3                 m4\n");
    printf("  %.17e  %.17e  %.17e  %.17e\n\n",
           (double)mu_det_r, (double)var_det, (double)mu3_det, (double)mu4_det);

    printf("# φ(x) = cos | α = %.3f | ρ stats after %d steps (self-cov):\n",
           (double)alpha, STEPS);
    printf("#   mean               var                m3                 m4\n");
    printf("  %.17e  %.17e  %.17e  %.17e\n\n",
           (double)mu_self_r, (double)var_self, (double)mu3_self, (double)mu4_self);

    char dtstr[16]; snprintf(dtstr,sizeof(dtstr),"%.0e",dt);
    for(int mode=0; mode<2; ++mode){
        double *s1 = mode? s1_self: s1_det;
        double *s2 = mode? s2_self: s2_det;
        double *s3 = mode? s3_self: s3_det;
        double *s4 = mode? s4_self: s4_det;
        double *mean = mode? mean_self: mean_det;
        char tag[8]; snprintf(tag,sizeof(tag), mode? "self":"det");
        char fname[256];
        snprintf(fname,sizeof(fname),"stats_%s_dt%s_steps%d_loop%d.txt",
                 tag, dtstr, STEPS, LOOP);
        FILE *fp=fopen(fname,"w");
        fprintf(fp,"# x   mean   var   m3   m4\n");
        for(int j=0;j<N;++j){
            double mu  = mean[j];
            double n   = (double)LOOP;
            double v2  = s2[j]/n - mu*mu;
            double v3  = s3[j]/n - 3*mu*(s2[j]/n) + 2*pow(mu,3);
            double v4  = s4[j]/n - 4*mu*(s3[j]/n) + 6*mu*mu*(s2[j]/n)
                                   - 3*pow(mu,4);
            fprintf(fp,"%.8e %.8e %.8e %.8e %.8e\n",
                    (double)(j*dx), mu, v2, v3, v4);
        }
        fclose(fp);
        printf("Wrote %s\n", fname);
    }

    for(int mode=0; mode<2; ++mode){
        const char* covtag = mode? "self":"det";
        long double (*S1)[NPHI] = mode? S1_self : S1_det;
        long double (*S2)[NPHI] = mode? S2_self : S2_det;
        long double (*S3)[NPHI] = mode? S3_self : S3_det;
        long double (*S4)[NPHI] = mode? S4_self : S4_det;

        for(int p=0;p<NPHI;++p){
            char fname[512];
            snprintf(fname,sizeof(fname),
                "phi_%s_moments_Z_%s_dt%s_steps%d_loop%d.txt",
                phi_tag[p], covtag, dtstr, STEPS, LOOP);
            FILE *fp=fopen(fname,"w");
            fprintf(fp,"# t_step   t_time   mean(Z)    var(Z)     c3(Z)      c4(Z)      sem(Z)   (Z=<%s,u(t)>, cov=%s)\n",
                    phi_tag[p], covtag);

            printf("== Moments via Z=<%s,u> [%s] ==\n", phi_tag[p], covtag);
            printf("step  time      mean(Z)            var(Z)             c3(Z)               c4(Z)               sem(Z)\n");

            for(int m=0;m<NCHK;++m){
                long double n = (long double)LOOP;
                long double mu = S1[m][p]/n;
                long double m2 = S2[m][p]/n;
                long double m3 = S3[m][p]/n;
                long double m4 = S4[m][p]/n;
                long double v2 = m2 - mu*mu;
                if (v2 < 0.0L) v2 = 0.0L;
                long double c3 = m3 - 3.0L*mu*m2 + 2.0L*mu*mu*mu;
                long double c4 = m4 - 4.0L*mu*m3 + 6.0L*mu*mu*m2 - 3.0L*mu*mu*mu*mu;
                long double sem = sqrtl(v2) / sqrtl((long double)LOOP);

                int step = tchk[m];
                double timeT = step*dt;

                fprintf(fp,"%6d  %.8e  %.17e  %.17e  %.17e  %.17e  %.17e\n",
                        step, timeT, (double)mu, (double)v2, (double)c3, (double)c4, (double)sem);

                printf("%4d  %.5g  %.17e  %.17e  %.17e  %.17e  %.17e\n",
                       step, timeT, (double)mu, (double)v2, (double)c3, (double)c4, (double)sem);
            }
            fclose(fp);
            printf("Wrote %s\n\n", fname);
            fflush(stdout);
        }
    }

    /* Output the moments of u(x,t) at each checkpoint to text */
    for(int mode=0; mode<2; ++mode){
        const char* covtag = mode? "self":"det";
        long double (*A1)[N] = mode? U1_self : U1_det;
        long double (*A2)[N] = mode? U2_self : U2_det;
        long double (*A3)[N] = mode? U3_self : U3_det;
        long double (*A4)[N] = mode? U4_self : U4_det;

        for(int m=0;m<NCHK;++m){
            char fname[512];
            snprintf(fname,sizeof(fname),
                     "density_moments_step%06d_%s_dt%s_steps%d_loop%d.txt",
                     tchk[m], covtag, dtstr, STEPS, LOOP);

            FILE *fp = fopen(fname,"w");
            if(!fp){
                fprintf(stderr,"failed to open %s\n", fname);
                exit(EXIT_FAILURE);
            }

            fprintf(fp,
                    "# step %d  time %.17e  cov=%s\n",
                    tchk[m], tchk[m]*dt, covtag);
            fprintf(fp,
                    "# x   mean(u)   var(u)   c3(u)   c4(u)   sem(u)\n");

            for(int j=0;j<N;++j){
                long double n = (long double)LOOP;
                long double mu = A1[m][j]/n;
                long double m2 = A2[m][j]/n;
                long double m3 = A3[m][j]/n;
                long double m4 = A4[m][j]/n;
                long double v2 = m2 - mu*mu;
                if (v2 < 0.0L) v2 = 0.0L;
                long double c3 = m3 - 3.0L*mu*m2 + 2.0L*mu*mu*mu;
                long double c4 = m4 - 4.0L*mu*m3 + 6.0L*mu*mu*m2 - 3.0L*mu*mu*mu*mu;
                long double sem = sqrtl(v2) / sqrtl((long double)LOOP);

                fprintf(fp,"%.17e %.17e %.17e %.17e %.17e %.17e\n",
                        x[j], (double)mu, (double)v2, (double)c3, (double)c4, (double)sem);
            }

            fclose(fp);
            printf("Wrote %s\n", fname);
        }
    }

    /* Save the representative density (loop=0) and mean density at each checkpoint */
    for(int mode=0; mode<2; ++mode){
        const char *covtag = mode ? "self" : "det";
        double *Urep = mode ? Urep_self : Urep_det;
        long double (*A1)[N] = mode ? U1_self : U1_det;

        for(int m=0; m<NCHK; ++m){
            char fname_rep[512];
            char fname_mean[512];

            snprintf(fname_rep, sizeof(fname_rep),
                     "density_representative_step%06d_%s_dt%s_steps%d_loop%d.txt",
                     tchk[m], covtag, dtstr, STEPS, LOOP);

            snprintf(fname_mean, sizeof(fname_mean),
                     "density_mean_step%06d_%s_dt%s_steps%d_loop%d.txt",
                     tchk[m], covtag, dtstr, STEPS, LOOP);

            /* representative: loop=0 */
            {
                FILE *fp = fopen(fname_rep, "w");
                if(!fp){
                    fprintf(stderr, "failed to open %s\n", fname_rep);
                    exit(EXIT_FAILURE);
                }
                fprintf(fp, "# representative density profile at step=%d time=%.17e cov=%s\n",
                        tchk[m], tchk[m]*dt, covtag);
                fprintf(fp, "# representative realization = loop 0\n");
                fprintf(fp, "# x u\n");
                size_t base = (size_t)m * (size_t)N;
                for(int j=0;j<N;++j){
                    fprintf(fp, "%.17e %.17e\n", x[j], Urep[base + (size_t)j]);
                }
                fclose(fp);
                printf("Wrote %s\n", fname_rep);
            }

            /* mean over loops */
            {
                FILE *fp = fopen(fname_mean, "w");
                if(!fp){
                    fprintf(stderr, "failed to open %s\n", fname_mean);
                    exit(EXIT_FAILURE);
                }
                fprintf(fp, "# mean density profile at step=%d time=%.17e cov=%s\n",
                        tchk[m], tchk[m]*dt, covtag);
                fprintf(fp, "# averaged over LOOP=%d realizations\n", LOOP);
                fprintf(fp, "# x mean_u\n");
                for(int j=0;j<N;++j){
                    long double mu = A1[m][j] / (long double)LOOP;
                    fprintf(fp, "%.17e %.17e\n", x[j], (double)mu);
                }
                fclose(fp);
                printf("Wrote %s\n", fname_mean);
            }
        }
    }

    /* ---- Mean profile ---- */
    char fname[256];
    snprintf(fname,sizeof(fname),
             "mean_profile_det_dt%s_steps%d_loop%d.txt",
             dtstr, STEPS, LOOP);
    FILE *f=fopen(fname,"w");
    for(int i=0;i<N;++i) fprintf(f,"%.8e %.8e\n",(double)(i*dx),mean_det[i]);
    fclose(f);
    printf("Wrote %s\n",fname);

    snprintf(fname,sizeof(fname),
             "mean_profile_self_dt%s_steps%d_loop%d.txt",
             dtstr, STEPS, LOOP);
    f=fopen(fname,"w");
    for(int i=0;i<N;++i) fprintf(f,"%.8e %.8e\n",(double)(i*dx),mean_self[i]);
    fclose(f);
    printf("Wrote %s\n",fname);

   
    free(x); free(phi);
    free(u_init_analytic); free(cdf_init_analytic);
    free(rho_det); free(rho_self);
    free(mean_det); free(mean_self);
    free(s1_det); free(s2_det); free(s3_det); free(s4_det);
    free(s1_self); free(s2_self); free(s3_self); free(s4_self);
    free(Urep_det);
    free(Urep_self);
    for(int p=0;p<NPHI;++p) free(phi_set[p]);

#if STORE_Z_SAMPLES
    free(Z_det_samples);
    free(Z_self_samples);
#endif

#if STORE_U_SAMPLES
    free(U_det_samples);
    free(U_self_samples);
#endif

    return 0;
}
/* --------------------  End of file  -------------------- */

// bv = b-vector (mesurement data), bval = b-values or TE's, xv = results-vector of prefactors, tv = x-values of xv
// N = No. of measurements, M = No. of exponentials, fmin = ADC/T2 minimum, fmax = ADC/T2 maximum
extern __declspec(dllexport) int DecayAna(float *bv, float *bval, float *xv, 
    float *tv, int N, int M, float fmin, float fmax, bool bRange, float *fRange,
    int max_iter, float mu, bool bReg, int iOrder)
{
    //Beep(700, 40);
    // generate M LogSpaced numbers between fmin and fmax
    float *v;
    v = new float[M];
    GenerateLogSpacedNumbers(fmin, fmax, M, bRange, fRange, v);
    // give back tv
    for (int i = 0; i < M; i++) {
        tv[i] = v[i];
    }
    // create vector b
    if (!bReg) {
        VectorXd b(N);
        for (int i = 0; i < N; i++) {
            b(i) = bv[i]; // / 20.0; //divide by noise?
        }
        MatrixXd A(N, M);
        for (int i = 0; i < N; i++) {   // exponentials
            for (int j = 0; j < M; j++) {
                A(i, j) = exp(-v[j] * bval[i]);// / 20.0; //divide by noise
            }
        }
        // results vector
        VectorXd x(M);
        // run nnls
        NNLS<MatrixXd>::solve(A, b, x, max_iter, 1e-10);
        // give back results
        for (int i = 0; i < M; i++) {
            xv[i] = (float)x(i);
        }
    }
    else
    {
        VectorXd b(N + M);
        for (int i = 0; i < N + M; i++) {
            if (i < N)
                b(i) = bv[i] / 20.0; //divide by noise?
            else
                b(i) = 0.0;
        }
        MatrixXd A(N + M, M);
        //for (int i = 0; i < N; i++) { // x-values - why should this be the x-values?
        //  A(i, 0) = bval[i];
        //}
        for (int i = 0; i < N + M; i++) {   // exponentials
            for (int j = 0; j < M; j++) {
                //for (int j = 1; j < M; j++) {
                if (i < N)
                    A(i, j) = exp(-v[j] * bval[i]) / 20.0; //divide by noise
                else {
                    if (iOrder == 0) {
                        A(i, j) = 0.0;
                        if (i - N == j)
                            A(i, j) = 1.0*mu;
                    }
                    if (iOrder == 1) {
                        A(i, j) = 0.0;
                        if (i - N == j)
                            A(i, j) = -1.0*mu; // -2.0;
                        if (i - N == (j + 1))
                            A(i, j) = 1.0*mu; // 1.0;
                    }
                    if (iOrder == 2) {
                        A(i, j) = 0.0;
                        if (i - N == j)
                            A(i, j) = -2.0*mu; // -2.0;
                        if (i - N == (j + 1))
                            A(i, j) = 1.0*mu; // 1.0;
                        if (i - N == (j - 1))
                            A(i, j) = 1.0*mu; // 1.0;
                    }
                    if (iOrder == 3) {  
                        A(i, j) = 0.0;
                        if (i - N == j)     // original 3rd order
                            A(i, j) = -6.0*mu; // -2.0;
                        if (i - N == (j + 1))
                            A(i, j) = 2.0*mu; // 1.0;
                        if (i - N == (j - 1))
                            A(i, j) = 2.0*mu; // 1.0;
                        if (i - N == (j + 2))
                            A(i, j) = 1.0*mu; // 1.0;
                        if (i - N == (j - 2))
                            A(i, j) = 1.0*mu; // 1.0;   
                        // new constrained regularization iwth prior knowledge from ROI
                        // 2nd order regularization
                        // seems useless... what did DeLuca et al. NMR Biomed 2018;31:e3965 ???
                        /*if (i - N == j)
                            A(i, j) = -2.0*mu; // -2.0;
                        if (i - N == (j + 1))
                            A(i, j) = 1.0*mu; // 1.0;
                        if (i - N == (j - 1))
                            A(i, j) = 1.0*mu; // 1.0;*/
                    }
                    /*A(i, j) = 0.0;
                    if (i - N == j)
                        A(i, j) = -2.0*mu; // -2.0;
                    if (i - N == (j + 1))
                        A(i, j) = 1.0*mu; // 1.0;
                    if (i - N == (j - 1))
                        A(i, j) = 1.0*mu; // 1.0;*/
                }
                /*char *ctr;
                ctr = new char[5];
                sprintf(ctr, "%2f\n", A(i, j));
                OutputDebugStringA(ctr);*/
            }
        }
        // new constrained regularization iwth prior knowledge from ROI
        // 2nd order regularization
        // seems useless... what did DeLuca et al. NMR Biomed 2018;31:e3965 ???
/*      for (int i = 0; i < N + M; i++) {   // exponentials
            for (int j = 0; j < M; j++) {
                if (iOrder == 3) {
                    // prior knowledge
                    if (i - N == j && j == 0) {
                        A(i, j) = 314.1*mu;
                        A(i, j + 1) = -314.1 / 2.0*mu;
                    }
                    if (i - N == j && j == 57) {
                        A(i, j) = 186.5*mu;
                        A(i, j - 1) = -186.5 / 2.0*mu;
                        A(i, j + 1) = -186.5 / 2.0*mu;
                    }
                    if (i - N == j && j == 140) {
                        A(i, j) = 32.5*mu;
                        A(i, j - 1) = -32.5 / 2.0*mu;
                        A(i, j + 1) = -32.5 / 2.0*mu;

                    }
                    if (i - N == j && j == 140) {
                        A(i, j) = 2.2*mu;
                        A(i, j + 1) = -2.2 / 2.0*mu;
                        A(i, j - 1) = -2.2 / 2.0*mu;
                    }
                }
            }
        }*/
        // results vector
        VectorXd x(M);
        // run nnls
        NNLS<MatrixXd>::solve(A, b, x, max_iter, 1e-10);
        // give back results
        for (int i = 0; i < M; i++) {
            xv[i] = (float)x(i);
        }
    }   
    // clean up 
    delete []v;
    return 0;
    /*VectorXd x(M);
    A << 1, 1, 1,
        2, 4, 8,
        3, 9, 27,
        4, 16, 64;
    b << 0.73, 3.24, 8.31, 16.72;
    // drive nnls
    NNLS<MatrixXd>::solve(A, b, x, max_iter, 1e-10);*/
}
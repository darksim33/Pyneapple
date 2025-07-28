from pyneapple import NNLSResults


def test_nnls_eval_fitting_results(nnls_fit_results, nnls_params, seg_reduced):
    fit_results = NNLSResults(nnls_params)
    fit_results.eval_results(nnls_fit_results[0])
    for idx in nnls_fit_results[3]:
        assert fit_results.f[idx].all() == nnls_fit_results[2][idx].all()
        assert fit_results.d[idx].all() == nnls_fit_results[1][idx].all()


def test_nnls_apply_auc(nnls_params, nnls_fit_results, seg_reduced):
    fit_results = NNLSResults(nnls_params)
    fit_results.eval_results(nnls_fit_results[0])
    assert nnls_params.apply_AUC_to_results(fit_results)

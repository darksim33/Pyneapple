def test_import_gpufit():
    import pygpufit

    assert pygpufit is not None


def test_cuda_check():
    import pygpufit.gpufit as gpufit

    gpufit.cuda_available()
    assert True

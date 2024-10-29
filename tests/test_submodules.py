from radimgarray import RadImgArray

def test_radimgarray_init():
    # Test that the RadImgArray class can be initialized
    array = RadImgArray([])
    assert array is not None

import numpy as np

from pyneapple.results.result_dict import ResultDict
from radimgarray import RadImgArray


def test_get():
    result_dict = ResultDict()
    result_dict[(1, 1, 1)] = 1.1

    assert result_dict[(1, 1, 1)] == 1.1
    assert result_dict.get((1, 1, 1), 0) == 1.1


def test_get_seg():
    result_dict = ResultDict()
    result_dict[(1, 1, 1)] = 1.1
    result_dict.set_segmentation_wise({(1, 1, 1): 1})
    result_dict[1] = 1.1

    assert result_dict[(1, 1, 1)] == 1.1
    assert result_dict.get((1, 1, 1), 0) == 1.1


def test_validate_key():
    result_dict = ResultDict()
    # result_dict[(1, 1, 1)] = 1.1
    # result_dict.set_segmentation_wise({(1, 1, 1): 1})
    result_dict[np.int32(1)] = 1.1
    for key in result_dict:
        assert isinstance(key, int)


def test_update():
    values_1 = {(1, 1, 1): 1.5}
    values_2 = {(0, 0, 0): 1.7}
    result_dict = ResultDict()
    result_dict.update(values_1)
    assert result_dict[(1, 1, 1)] == 1.5
    result_dict.update(values_2)
    assert result_dict[(0, 0, 0)] == 1.7


def test_as_array():
    result_dict = ResultDict()
    test_array = np.array([[1.3, 2.2], [3.1, 4.4]])
    result_dict.update(
        {
            (0, 0): test_array[0, 0],
            (0, 1): test_array[0, 1],
            (1, 0): test_array[1, 0],
            (1, 1): test_array[1, 1],
        }
    )
    array = result_dict.as_array((2, 2, 1, 1))
    assert np.allclose(array, test_array[..., np.newaxis, np.newaxis])


def test_as_rad_img_array():
    result_dict = ResultDict()
    test_array = RadImgArray(np.array([[1.3, 2.2], [3.1, 4.4]]))
    result_dict.update(
        {
            (0, 0, 0, 0): test_array[0, 0],
            (0, 1, 0, 0): test_array[0, 1],
            (1, 0, 0, 0): test_array[1, 0],
            (1, 1, 0, 0): test_array[1, 1],
        }
    )
    test_array = test_array[:, :, np.newaxis, np.newaxis]
    array = result_dict.as_RadImgArray(test_array)
    assert np.allclose(array, test_array)

from __future__ import annotations

from .. import IDEALParams
from ..utils.logger import logger
from radimgarray import RadImgArray, SegImgArray


def fit(img: RadImgArray, params: IDEALParams, fit_type: str = None, **kwargs):
    results = np.zeros(1, 1, dtype=np.float32)
    for idx, step in enumerate(params.dim_steps):
        logger.info(f"Fitting {step.name}")
        _img = params.interpolate_img(img, step)
        _seg = params.interpolate_seg(seg, step)
        x0, lb, ub = params.get_boundaries(results)
        pixel_args = params.get_pixel_args(_img, _seg, x0, lb, ub)
        step_results = params.fit_handler(pixel_args, fit_type, **kwargs)

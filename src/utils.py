import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path

# from PyQt6.QtGui import QPixmap
import imantics


# v0.1


class Nii:
    """
    Class based on NiBabels NifTi-class with additional functionality

    ...

    Attributes
    ----------
    path : Path
        Path to nifti file
    array : np.ndarray
        Image array
    affine : np.ndarray
        Image Rotation Matrix
    header : nib.Nifti1Header()
        NifTi header information
    # mask : bool
    #     Mask or anatomical image

    Methods
    ----------
    load(path=None)
        Load NifTi image
    save()
        Save to NifTi image
    reset()
        Reload NifTi image
    copy()
        copy Nii
    show(slice)
        Show image of selected Slice as PNG
    from_array(array)
        Create NifTi object from np.ndarray
    to_rgba_array(slice, alpha)
        Return RGBA ndarray
    to_rgba_image(slice, alpha)
        Return RGBA PIL.Image
    QPixmap(slice, scaling)
        Return QPixmap from Slice

    """

    def __init__(self, path: str | Path | None = None, **kwargs) -> None:
        self._original_scaling = None
        self.path = path
        # self.__set_path(path)
        self.array = np.zeros((1, 1, 1, 1))
        self.affine = np.eye(4)
        self.header = nib.Nifti1Header()
        # self.mask: bool = False
        self.__load()
        for key in kwargs:
            if key == "do_zero_padding" and kwargs["do_zero_padding"]:
                self.zero_padding()

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, value: str | Path | None):
        if isinstance(value, str):
            value = Path(value)
        # elif value is None:
        #     # value = Path.cwd() / ".temp.nii" # for handling of the save process
        #     pass
        self._path = value

    def load(self, path: Path | str):
        """Load NifTi file."""
        self.__load()

    def __load(self) -> None:
        """Private Loader"""
        if self.path is None:
            return None
        if self.path.is_file():
            nifti = nib.load(self.path)
            self.array = np.array(nifti.get_fdata())
            while len(self.array.shape) < 4:
                self.array = np.expand_dims(self.array, axis=-1)
            self.affine = nifti.affine
            self.header = nifti.header
        else:
            print("File not found!")
            return None

    def reset(self):
        """Resets Nii by loading the file again"""
        self.__load()

    def clear(self):
        """Removes all data from obj"""

        self.path = None
        self.array = np.zeros((1, 1, 1, 1))
        self.affine = np.eye(4)
        self.header = nib.Nifti1Header()
        # self.mask: bool = False

    def zero_padding(self):
        if self.array.shape[0] < self.array.shape[1]:
            new_array = np.pad(
                self.array,
                (
                    (
                        int((self.array.shape[1] - self.array.shape[0]) / 2),
                        int((self.array.shape[1] - self.array.shape[0]) / 2),
                    ),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                ),
                mode="constant",
            )
            self.array = new_array
        elif self.array.shape[1] < self.array.shape[0]:
            new_array = np.pad(
                self.array,
                (
                    (0, 0),
                    (
                        int((self.array.shape[0] - self.array.shape[1]) / 2),
                        int((self.array.shape[0] - self.array.shape[1]) / 2),
                    ),
                    (0, 0),
                    (0, 0),
                ),
                mode="constant",
            )
            self.array = new_array
        self.header.set_data_shape(self.array.shape)

    def save(self, name: str | Path, dtype: object = int, do_zip: bool = True):
        """
        Save Nii to File

        Attributes:
            name (str|Path): Name of the output file to save the data to.
            dtype (object): Sets the output data type of the NifTi (int and float supported)
            do_zip (bool): "Will force the zipping of the file
        """

        save_path = self.path.parent / name if self.path is not None else name
        # Save as Int/float
        array = np.array(self.array.astype(dtype).copy())
        header = self.header
        if dtype == int:
            header.set_data_dtype("i4")
        elif dtype == float:
            header.set_data_dtype("f4")
        new_Nii = nib.Nifti1Image(
            array,
            self.affine,
            header,
        )
        # https://note.nkmk.me/en/python-numpy-dtype-astype/
        # https://brainder.org/2012/09/23/the-nifti-file-format/
        nib.save(new_Nii, save_path)

    def copy(self):
        """Make Copy of Nii class obj"""

        return deepcopy(self)

    def from_array(
        self,
        array: np.ndarray,
        header: nib.Nifti1Header | None = None,
        path: str | Path | None = None,
    ):
        """Create Nii image with a given or default header"""

        self.array = array
        self.affine = np.eye(4)
        self.header = nib.Nifti1Header() if not header else header
        self.path = path
        return self

    def to_rgba_array(self, slice_number: int = 0, alpha: int = 1) -> np.ndarray:
        """Return RGBA array"""

        # Return RGBA array of Nii
        # rot Image
        array = (
            np.rot90(self.array[:, :, slice_number, 0])
            if slice_number is not None
            else self.array[:, :, 0, 0]
        )
        # Add check for empty mask
        array_norm = (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
        alpha_map = np.ones(array_norm.shape)
        array_rgba = np.dstack((array_norm, array_norm, array_norm, alpha_map))

        return array_rgba

    # Might be unnecessary by now. Only works with plotting.overlay_image
    def to_rgba_image(self, slice: int = 0, alpha: int = 1) -> Image:
        # Return RGBA PIL Image of Nii slice
        array_rgba = self.to_rgba_array(slice, alpha) * 255
        img_rgba = Image.fromarray(
            array_rgba.round().astype(np.int8).copy(),  # Needed for Image
            "RGBA",
        )

        return img_rgba

    def show(self, slice: int | None = None, tag: str = "array"):
        """Show image as matplotlib figure or PNG"""

        if tag == "png" or tag == "image":
            img_rgb = self.to_rgba_image(slice)
            img_rgb.show()
        elif tag == "array":
            array_rgb = self.to_rgba_array(slice)
            fig, ax = plt.subplots()
            ax.imshow(array_rgb[:, :, :])
            plt.set_cmap("gray")
            plt.show()

    def scale_image(self, scaling: str | None | int = None):
        """
        Scale image.

        Parameters:
            scaling (str): Scaling option. Options are "S/S0" or a scalar
        """
        if scaling == "S/S0":
            array = self.array.copy()
            s0 = array[:, :, :, 0]
            self._original_scaling = s0
            new_array = np.divide(array, s0[:, :, :, np.newaxis])
            self.array = new_array
        elif isinstance(scaling, int):
            self.array = self.array * scaling

    # Might be unnecessary by now
    # def QPixmap(self, slice: int = 0, scaling: int = 1) -> QPixmap:
    #     if self.path:
    #         img = self.to_rgba_image(slice).copy()
    #         img = img.resize(
    #             [img.size[0] * scaling, img.size[1] * scaling],
    #             Image.NEAREST,
    #         )
    #         qPixmap = QPixmap.fromImage(ImageQt.ImageQt(img))
    #         return qPixmap
    #     else:
    #         return None


class NiiSeg(Nii):
    """
    Nii segmentation image: can be a mask or a ROI based nifti image.

    slices_contain_seg: Puts out boolean array for all slices indicating if segmentation is present
    """

    def __init__(self, path: str | Path | None = None):
        self.path = None
        super().__init__(path)
        # check segmentation dimension
        if len(self.array.shape) > 3:
            self.array = self.array[..., :1]
        self._seg_indexes = None
        self.mask = True
        self.slices_contain_seg = np.any(self.array != 0, axis=(0, 1))
        self._n_segmentations = (
            np.unique(self.array).max() if self.path is not None else None
        )
        self.segmentations = None
        self.calculate_polygons()

    def from_array(
        self,
        array: np.ndarray,
        header: nib.Nifti1Header | None = None,
        path: str | Path | None = None,
    ):
        """Create Nii image with a given or default header"""

        self.array = array
        self.affine = np.eye(4)
        self.header = nib.Nifti1Header() if not header else header
        self.path = path
        self.calculate_polygons()
        return self

    def clear(self):
        super().clear()
        self.calculate_polygons()

    @property
    def n_segmentations(self) -> np.ndarray:
        """Number of Segmentations"""

        if self.path:
            self._n_segmentations = np.unique(self.array).max()
        return self._n_segmentations.astype(int)

    @property
    def seg_indexes(self) -> np.ndarray:
        if self.path:
            seg_indexes = np.unique(self.array)
            if seg_indexes[0] == 0:
                seg_indexes = seg_indexes[1:]
            self._seg_indexes = seg_indexes
        elif not self.path and not self._seg_indexes:
            self._seg_indexes = None
        return self._seg_indexes

    def calculate_polygons(self):
        """
        Lists polygons/segmentations for all slices.

        Creates a list containing one list for each slice (size = number of slices).
        Each of these lists contains the lists for each segmentation (number of segmentations).
        Each of these lists contains the polygons(/segmentation obj?) found in that slice (this length might be
        varying)
        """

        if self.path:
            segmentations = dict()
            for seg_index in self.seg_indexes:
                segmentations[seg_index] = Segmentation(self.array, seg_index)
            self.segmentations = segmentations

    @staticmethod
    def evaluate_seg():
        print("Evaluating Segmentation")

    def get_seg_index_positions(self, seg_index: int) -> list:
        # might be removed (unused)
        """
        The get_seg_index_positions function takes a segmentation index as input and returns the positions of all voxels with that index.

        Parameters
        ----------
            self
                Represent the instance of the class
            seg_index
                Find the indexes of a specific segmentation index

        Returns
        -------

            A list of the positions of all voxels with a certain segmentation index
        """

        idxs_raw = np.array(np.where(self.array == seg_index))
        idxs = list()
        for idx in range(len(idxs_raw[0])):
            idxs.append(
                [idxs_raw[0][idx], idxs_raw[1][idx], idxs_raw[2][idx], idxs_raw[3][idx]]
            )
        return idxs

    def get_seg_coordinates(self, seg_index: int | str) -> list | None:
        """
        Return voxel positions of non-zero segmentation.

        seg_index: int | str
            Can either be a selected index or a string.
            Allowed strings are "nonzero".
        """
        if isinstance(seg_index, str):
            if seg_index == "nonzero":
                nonzero_indices = np.nonzero(self.array)
                coordinates = list(
                    zip(nonzero_indices[0], nonzero_indices[1], nonzero_indices[2])
                )
                return coordinates
            else:
                raise Exception(
                    "Invalid segment keyword or index. Only strings (nonzero) are allowed!"
                )
        elif isinstance(seg_index, (int, np.int8, np.int16, np.int32)):
            indices = np.where(self.array == seg_index)
            coordinates = list(zip(indices[0], indices[1], indices[2]))
            return coordinates

    def get_array_for_seg(self, seg_index: int | str) -> np.ndarray:
        array = np.zeros(self.array.shape)
        np.put(array, self.get_seg_coordinates(seg_index), 1)
        return array

    def to_rgba_array(self, slice_number: int = 0, alpha: int = 1) -> np.ndarray:
        """Return RGBA array"""

        # TODO this might need some fixing the way that only the segmented areas get a alpha larger 0
        # Return RGBA array of Nii
        # rot Image
        array = (
            np.rot90(self.array[:, :, slice_number])
            if slice_number is not None
            else self.array[:, :, 0, 0]
        )
        # Add check for empty mask
        array_norm = (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
        # if nifti is mask -> Zeros get zero alpha
        alpha_map = array_norm * alpha  # if self.mask else np.ones(array.shape)
        if type(self) == NiiSeg:
            alpha_map[alpha_map > 0] = 1
        array_rgba = np.dstack((array_norm, array_norm, array_norm, alpha_map))

        return array_rgba


class Segmentation:
    def __init__(self, seg_img: np.ndarray, seg_index: int):
        self.seg_index = seg_index
        self.img = seg_img.copy()
        # check if image contains only the selected seg_index else change
        self.img[self.img != seg_index] = 0
        self.polygons = dict()
        self.polygon_patches = dict()
        self.__get_polygons()
        # self.number_polygons = len(self.polygons)

    def __get_polygons(self):
        """
        Create imantics Polygon list of image array.

        The __get_polygons function is a helper function that uses the imantics library to convert
        the image array into a list of Polygon objects. The polygons are stored in self.polygons, and
        a list of patches for each slice is stored in self.polygon_patches.

        Parameters
        ----------
            self
                Refer to the current object

        Returns
        -------

            A list of polygon objects

        Doc Author
        ----------
            Trelent
        """

        # Set dictionaries for polygons
        polygons = dict()
        polygon_patches = dict()

        for slice_number in range(self.img.shape[2]):
            polygons[slice_number] = (
                imantics.Mask(np.rot90(self.img[:, :, slice_number])).polygons()
                if not None
                else None
            )
            # Transpose Points to fit patchify
            points = list()
            for poly in polygons[slice_number].points:
                points.append(poly.T)
            if len(points):
                if len(points) > 1:
                    polygon_patches[slice_number] = [self.patchify(points)]
                else:
                    polygon_patches[slice_number] = [
                        patches.Polygon(polygons[slice_number].points[0])
                    ]
        self.polygons = polygons
        self.polygon_patches = polygon_patches

    # https://gist.github.com/yohai/81c5854eaa4f8eb5ad2256acd17433c8
    @staticmethod
    def patchify(polys):
        """
        Returns a matplotlib patch representing the polygon with holes.

        polys is an iterable (i.e. list) of polygons, each polygon is a numpy array
        of shape (2, N), where N is the number of points in each polygon. The first
        polygon is assumed to be the exterior polygon and the rest are holes. The
        first and last points of each polygon may or may not be the same.
        This is inspired by
        https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
        Example usage:
        ext = np.array([[-4, 4, 4, -4, -4], [-4, -4, 4, 4, -4]])
        t = -np.linspace(0, 2 * np.pi)
        hole1 = np.array([2 + 0.4 * np.cos(t), 2 + np.sin(t)])
        hole2 = np.array([np.cos(t) * (1 + 0.2 * np.cos(4 * t + 1)),
                          np.sin(t) * (1 + 0.2 * np.cos(4 * t))])
        hole2 = np.array([-2 + np.cos(t) * (1 + 0.2 * np.cos(4 * t)),
                          1 + np.sin(t) * (1 + 0.2 * np.cos(4 * t))])
        hole3 = np.array([np.cos(t) * (1 + 0.5 * np.cos(4 * t)),
                          -2 + np.sin(t)])
        holes = [ext, hole1, hole2, hole3]
        patch = patchify([ext, hole1, hole2, hole3])
        ax = plt.gca()
        ax.add_patch(patch)
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])
        """

        # TODO: this only works as desired if the first is the exterior and none of the other regions is outside the first one therefor the segmentation needs to be treated accordingly

        def reorder(poly, cw=True):
            """
            Reorders the polygon to run clockwise or counter-clockwise according to the value of cw.

            It calculates whether a polygon is cw or ccw by summing (x2-x1)*(y2+y1) for all edges of the polygon,
            see https://stackoverflow.com/a/1165943/898213.
            """
            # Close polygon if not closed
            if not np.allclose(poly[:, 0], poly[:, -1]):
                poly = np.c_[poly, poly[:, 0]]
            direction = (
                (poly[0] - np.roll(poly[0], 1)) * (poly[1] + np.roll(poly[1], 1))
            ).sum() < 0
            if direction == cw:
                return poly
            else:
                return np.array([p[::-1] for p in poly])

        def ring_coding(n):
            """
            Returns a list of len(n).

            Of this format:
            [MOVETO, LINETO, LINETO, ..., LINETO, LINETO CLOSEPOLY]
            """

            codes = [matplotlib.path.Path.LINETO] * n
            codes[0] = matplotlib.path.Path.MOVETO
            codes[-1] = matplotlib.path.Path.CLOSEPOLY
            return codes

        ccw = [True] + ([False] * (len(polys) - 1))
        polys = [reorder(poly, c) for poly, c in zip(polys, ccw)]
        path_codes = np.concatenate([ring_coding(p.shape[1]) for p in polys])
        vertices = np.concatenate(polys, axis=1)
        return patches.PathPatch(matplotlib.path.Path(vertices.T, path_codes))


class NiiFit(Nii):
    def __init__(
        self,
        path: str | Path | None = None,
        n_components: int | np.ndarray | None = 1,
        **kwargs,
    ):
        super().__init__(path, **kwargs)
        self.n_components = n_components
        self.d_weight = 10000
        self.f_weight = 100
        self.s0_weight = 1
        self.scaling = kwargs.get("scaling", None)

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, scale: np.ndarray | list | None = None):
        # TODO: Should also check for number of actually used components
        if scale is None:
            scaling = np.zeros(2 * self.n_components + 1)
            scaling[: self.n_components] = self.d_weight
            scaling[self.n_components : -1] = self.f_weight
            scaling[-1] = self.s0_weight
        elif isinstance(scale, np.ndarray):
            scaling = scale
        elif isinstance(scale, (list, tuple)):
            scaling = np.array(scale)
        else:
            scaling = None
        self._scaling = scaling

    def save(
        self,
        file_name: str | Path,
        dtype: object = int,
        save_type: str = "single",
        parameter_names: list | None = None,
        do_zip: bool = True,
    ) -> None:
        """
        Save array and save as int (float is optional but not recommended).

        Attributes:

        name: str
            File name of the saved file. (without parent Path)
        dtype: object
            Defines data type for nifti.
        save_type: str
            Defines what kind of Save is chosen.
            "single": all Data ist saved to a single file with the fourth dimension representing variables.

        Information:

        Nifti Header Codes:
            https://note.nkmk.me/en/python-numpy-dtype-astype/
            https://brainder.org/2012/09/23/the-nifti-file-format/
        """
        save_path = self.path.parent / file_name if self.path is not None else file_name
        if "gz" not in save_path.suffix and do_zip:
            save_path = save_path.with_suffix(save_path.suffix + ".gz")
        if save_type == "single":
            array = self.scale_image_all().astype(dtype)
            header = self.header
            if dtype == int:
                header.set_data_dtype("i4")
            elif dtype == float:
                header.set_data_dtype("f4")
            new_nii = nib.Nifti1Image(
                array,
                self.affine,
                header,
            )
            nib.save(new_nii, save_path)
        elif save_type == "separate":
            for comp in range(len(parameter_names)):
                array = self.scale_image_single_variable(
                    self.array[:, :, :, comp], self.scaling[comp]
                ).astype(dtype)
                header = self.header
                if dtype == int:
                    header.set_data_dtype("i4")
                elif dtype == float:
                    header.set_data_dtype("f4")
                new_nii = nib.Nifti1Image(
                    array,
                    self.affine,
                    header,
                )
                if parameter_names:
                    var_name = parameter_names[comp]
                else:
                    var_name = comp
                save_path_new = (
                    file_name.parent / f"{file_name.stem}_{var_name}{file_name.suffix}"
                )
                if "gz" not in save_path_new.suffix and do_zip:
                    save_path_new = save_path_new.with_suffix(
                        save_path_new.suffix + ".gz"
                    )
                print(f"Saving to: {save_path_new}")
                nib.save(new_nii, save_path_new)

    def scale_image_all(self) -> np.ndarray | None:
        """Scale all variables to clinical dimensions"""
        array = self.array.copy()
        if isinstance(self.n_components, int):
            scaling = np.zeros(2 * self.n_components + 1)
            scaling[: self.n_components] = self.d_weight
            scaling[self.n_components : -1] = self.f_weight
            scaling[-1] = self.s0_weight
            array_scaled = array * scaling
        elif isinstance(self.n_components, np.ndarray):
            array_scaled = None
        else:
            array_scaled = None
        return array_scaled

    @staticmethod
    def scale_image_single_variable(
        array: np.ndarray, scale: int | float | np.ndarray
    ) -> np.ndarray | None:
        """Scale a single variable to clinical dimensions"""
        if isinstance(scale, (int, float)):
            array_scaled = array * scale
        elif isinstance(scale, np.ndarray):
            array_scaled = None
        else:
            array_scaled = None
        return array_scaled


class NSlice:
    def __init__(self, value: int = None):
        if not value:
            self.__value = value
            self.__number = value + 1
        else:
            self.__value = None
            self.__number = None

    @property
    def number(self):
        return self.__number

    @property
    def value(self):
        return self.__value

    @number.setter
    def number(self, value):
        self.__number = value
        self.__value = value - 1

    @value.setter
    def value(self, value):
        self.__number = value + 1
        self.__value = value


class Processing(object):
    @staticmethod
    def merge_nii_images(img1: Nii | NiiSeg, img2: Nii | NiiSeg) -> Nii:
        """
        Takes two Nii or NiiSeg objects and returns a new Nii object.

        The function first checks if the input images are of type NiiSeg, and if so, it compares their in-plane sizes.
        If they match, then the function multiplies each voxel value in img2 by its corresponding voxel value in img2.
        This is done for every slice of both images (i.e., for all time points). The resulting array is assigned to a new
        Nii object which is returned by the function.
        """

        array1 = img1.array.copy()
        array2 = img2.array.copy()
        if type(img2) == NiiSeg:
            if np.array_equal(array1.shape[0:2], array2.shape[0:2]):
                # compare in plane size of Arrays
                array_merged = np.ones(array1.shape)
                for idx in range(img1.array.shape[3]):
                    array_merged[:, :, :, idx] = np.multiply(
                        array1[:, :, :, idx], array2[:, :, :, 0]
                    )
                img_merged = img1.copy()
                img_merged.array = array_merged
                return img_merged
        else:
            warnings.warn("Warning: Secondary Image is not a mask!")

    @staticmethod
    def get_mean_seg_signal(
        nii_img: Nii, nii_seg: NiiSeg, seg_index: int
    ) -> np.ndarray:
        img = nii_img.array.copy()
        seg_indexes = nii_seg.get_seg_index_positions(seg_index)
        number_of_b_values = img.shape[3]
        signal = np.zeros(number_of_b_values)
        for b_values in range(number_of_b_values):
            data = 0
            for idx in seg_indexes:
                idx[3] = b_values
                data = data + img[tuple(idx)]
            signal[b_values] = data / len(seg_indexes)
        return signal


class IndexTracker:
    def __init__(self, ax, X):
        self.index = 0
        self.X = X
        self.ax = ax
        self.im = ax.imshow(self.X[:, :, self.index], cmap="gray")
        self.update()

    def on_scroll(self, event):
        # print(event.button, event.step)
        increment = 1 if event.button == "up" else -1
        max_index = self.X.shape[-1] - 1
        self.index = np.clip(self.index + increment, 0, max_index)
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.index])
        self.ax.set_title(f"Use scroll wheel to navigate\nindex {self.index}")
        self.im.axes.figure.canvas.draw()

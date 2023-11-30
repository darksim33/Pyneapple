import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        self.path = None
        self.__set_path(path)
        self.array = np.zeros((1, 1, 1, 1))
        self.affine = np.eye(4)
        self.header = nib.Nifti1Header()
        # self.mask: bool = False
        self.__load()
        for key in kwargs:
            if key == "do_zero_padding" and kwargs["do_zero_padding"]:
                self.zero_padding()

    def load(self, path: Path | str):
        """Load NifTi file."""
        self.__set_path(path)
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

    def __set_path(self, path: str | Path):
        """Private Path setup"""
        self.path = Path(path) if path is not None else None

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
                    (0, (self.array.shape[1] - self.array.shape[0])),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                ),
                mode="constant",
            )
            self.array = new_array
        elif self.array.shape[0] < self.array.shape[1]:
            new_array = np.pad(
                self.array,
                (
                    (0, 0),
                    (0, (self.array.shape[1] - self.array.shape[0])),
                    (0, 0),
                    (0, 0),
                ),
                mode="constant",
            )
            self.array = new_array

    def save(self, name: str | Path, dtype: object = int):
        """Save Nii to File"""
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
    """Nii segmentation image: can be a mask or a ROI based nifti image
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

    @property
    def n_segmentations(self) -> np.ndarray:
        """Number of Segmentations"""
        if self.path:
            self._n_segmentations = np.unique(self.array).max()
        return self._n_segmentations.astype(int)

    @property
    def seg_indexes(self) -> list | None:
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
        Creates a list containing one list for each slice (size = number of slices).
        Each of these lists contains the lists for each segmentation (number of segmentations).
        Each of these lists contains the polygons(/segmentation obj?)
        find in that slice (this length might be varying)
        """
        if self.path:
            segmentations = dict()
            for seg_index in self.seg_indexes:
                segmentations[str(seg_index)] = Segmentation(self.array, seg_index)
            self.segmentations = segmentations

    # def get_single_seg_mask(self, number_seg: int):
    #     """Returns Nii_seg obj with only one segmentation"""
    #     seg = self.copy()
    #     seg_array = np.round(self.array.copy())
    #     seg_array[seg_array != number_seg] = 0
    #     seg.array = seg_array
    #     return seg

    def __get_polygon_patch_2d(
        self, number_seg: np.ndarray | int, number_slice: int
    ) -> list:  # list(imantics.annotation.Polygons):
        if number_seg <= self.n_segmentations.max():
            # Get array and set unwanted segmentation to 0
            seg = self.array.copy()
            seg_slice = np.round(np.rot90(seg[:, :, number_slice, 0]))
            seg_slice[seg_slice != number_seg] = 0

            if seg_slice.max() > 0:
                polygons = imantics.Mask(seg_slice).polygons()
                polygon_patches = [None] * len(polygons.polygons)
                for idx in range(len(polygons.polygons)):
                    polygon_patches[idx] = patches.Polygon(polygons.points[idx])
                    return polygon_patches

            # if seg_slice.max() > 0:
            #     polygon = self._get_polygon_of_slice(seg_slice)
            #     return patches.Polygon(polygon)
            # else:
            #     return None

    @staticmethod
    def evaluate_seg():
        print("Evaluating Segmentation")

    def get_seg_index_positions(self, seg_index):
        # might be removed (unused)
        idxs_raw = np.array(np.where(self.array == seg_index))
        idxs = list()
        for idx in range(len(idxs_raw[0])):
            idxs.append(
                [idxs_raw[0][idx], idxs_raw[1][idx], idxs_raw[2][idx], idxs_raw[3][idx]]
            )
        return idxs

    @staticmethod
    def __get_polygons_of_slice(seg: np.ndarray) -> imantics.Polygons:
        """Return imantics Polygon of image slice"""
        # polygon = list(Mask(seg).polygons().points[0])
        polygons = imantics.Mask(seg).polygons()
        return polygons

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
        self.polygons = list()
        self.polygon_patches = list()
        self.__get_polygons()
        # self.number_polygons = len(self.polygons)

    def __get_polygons(self):
        """Return imantics Polygon list of image array"""
        polygons = list()
        polygon_patches = list()
        for slice_number in range(self.img.shape[2]):
            polygon_array = (
                imantics.Mask(np.rot90(self.img[:, :, slice_number])).polygons()
                if not None
                else None
            )
            polygons.append(polygon_array)
            if polygon_array:
                slice_polygons = list()
                for polygon_idx in range(len(polygon_array.polygons)):
                    slice_polygons.append(
                        patches.Polygon(polygon_array.points[polygon_idx])
                    )
                polygon_patches.append(slice_polygons)
            else:
                polygons.append(None)
        self.polygons = polygons
        self.polygon_patches = polygon_patches


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

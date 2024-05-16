import numpy as np
from pathlib import Path
from datetime import datetime


class FreeDiffusionTool:
    def __init__(
        self,
        b_values: list | np.ndarray = np.array([0, 1000]),
        n_dims: int | None = 3,
        vendor: str | None = "Siemens_VE11c",
        **kwargs,
    ):
        self.b_values = b_values
        self.n_dims = n_dims
        self.vendor = vendor

        self.supported_vendors = [
            "Siemens_VE11c",
        ]

    @property
    def b_values(self):
        return self._b_values

    @b_values.setter
    def b_values(self, b_values):
        if isinstance(b_values, list):
            b_values = np.array(b_values)
        self._b_values = b_values

    @property
    def vendor(self):
        return self._vendor

    @vendor.setter
    def vendor(self, vendor):
        if vendor not in self.supported_vendors:
            raise ValueError(
                "The selected vendor is not supported. Check documentation for supported ones."
            )
        self._vendor = vendor

    def get_header(self, diffusion_vector_file):
        resource_path = Path(__file__).parent.parent / "resources" / "free_diffusion"
        file = self.find_header(resource_path, self.vendor)
    @staticmethod
    # def find_header(directory: Path | str, vendor: str):
    #     def find_substring_in_list(substring, string_list):
    #         return [s for s in string_list if substring in s]
    #
    #     if isinstance(directory, str):
    #         directory = Path(directory)
    #     file_list = []
    #     for idx in directory.iterdir():
    #         if idx.is_file():
    #             file_list.append(idx)
    #
    #     header_file = find_substring_in_list(vendor, file_list)
    #     return header_file
    #

    def get_diffusion_vectors(self) -> list:
        return list()

    def save_diffusion_vector_file(self, diffusion_vector_file: Path):
        if self.vendor == "Siemens_VE11c":
            self.write_siemens_ve11c(diffusion_vector_file)

    def write_siemens_ve11c(self, diffusion_vector_file: Path, **options: dict) -> None:

        def siemens_ve11c_header_constructor(**kwargs) -> list:
            head = list()
            now = datetime.now()

            current_time = now.strftime("%a %b %d %H:%M:%S %Y")

            # load optional settings
            filename = kwargs.get("filename", "MyVectorSet.dvs")
            if isinstance(filename, Path):
                filename = filename.name

            description = kwargs.get("description", "External vector file for SBBDiffusion")
            coordinate_system = kwargs.get("CoordinateSystem", "xyz")
            normalisation = kwargs.get("Normalisation", "maximum")
            comment = kwargs.get("Comment", "my diffusion vector set")

            head.append(
                r"# -----------------------------------------------------------------------------"
            )
            head.append(r"# Copyright (C) SIEMENS AG 2011 All Rights Reserved.\ ")
            head.append(
                r"# -----------------------------------------------------------------------------"
            )
            head.append(r"# ")
            head.append(r"# Project: NUMARIS/4")
            head.append(
                f"# File: c:\\Medcom\\MriCustomer\\seq\\DiffusionVectorSets\\{filename}"
            )
            head.append(f"# Date: {current_time}")
            head.append("#")
            head.append(f"Descrip: {description}")
            head.append(
                r"# -----------------------------------------------------------------------------"
            )
            head.append(r"[directions=%d]")
            head.append(f"CoordinateSystem = {coordinate_system}")
            head.append(f"Normalisation = {normalisation}")
            head.append(f"Comment = {comment}")
            return head

        header = siemens_ve11c_header_constructor(filename=diffusion_vector_file, **options)

        with diffusion_vector_file.open() as file:
            # write header to file
            for line in header:
                file.write(line)


    @staticmethod


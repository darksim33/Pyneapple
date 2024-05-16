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

        self.supported_vendors = [
            "Siemens_VE11c",
        ]

        self.vendor = vendor

    @property
    def vendor(self):
        """Handles different supported vendors."""
        return self._vendor

    @vendor.setter
    def vendor(self, vendor):
        if vendor not in self.supported_vendors:
            raise ValueError(
                "The selected vendor is not supported. Check documentation for supported ones."
            )
        self._vendor = vendor

    def get_diffusion_vectors(self) -> np.ndarray:
        """Calculate the diffusion vectors for the given number of dimensions and b_values."""
        diffusion_vectors = np.array([])

        # get equally spaced vectors
        phi = np.linspace(0, 2 * np.pi, self.n_dims)
        theta = np.linspace(0, np.pi / 2, self.n_dims)

        for b_value in self.b_values:
            # calculate vector of directions
            vectors = np.array(
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ]
            ).T

            # Apply b_values to vector file as relative length
            scaling = b_value / self.b_values[-1]

            if diffusion_vectors.size:
                diffusion_vectors = np.concatenate(
                    (diffusion_vectors, vectors * scaling)
                )
            else:
                diffusion_vectors = vectors * scaling

        return diffusion_vectors

    def save(self, diffusion_vector_file: Path):
        """Handles saving the diffusion vector file for different vendors."""
        if self.vendor == "Siemens_VE11c":
            self.write_siemens_ve11c(diffusion_vector_file)

    def write_siemens_ve11c(self, diffusion_vector_file: Path, **options: dict) -> None:
        """
        Write vector file for Siemens_VE11c.

        Parameters
        diffusion_vector_file: Path
            Pathlib Path to the diffusion vector file.

        options: dict
            Several options to modify the header information.
                description: str
                    Description of the diffusion vector file.
                CoordinateSystem: str = "xyz"
                    Coordinate System used by the scaner (?)
                Normalisation: str = "maximum"
                    Normalisation mode used by the scaner (?)
                Comment: str
                    Further information and comments about the diffusion vector file.

        """

        def construct_header(**kwargs) -> list:
            """
            Create a header string for the diffusion vector file.

            Options are explained in parent method documentation.
            """
            head = list()
            now = datetime.now()

            current_time = now.strftime("%a %b %d %H:%M:%S %Y")

            # load optional settings
            filename = kwargs.get("filename", "MyVectorSet.dvs")
            if isinstance(filename, Path):
                filename = filename.name

            description = kwargs.get(
                "description", "External vector file for SBBDiffusion"
            )
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
            head.append(f"# Descrip: {description}")
            head.append(
                r"# -----------------------------------------------------------------------------"
            )
            head.append(r"[directions=%d]")
            head.append(f"CoordinateSystem = {coordinate_system}")
            head.append(f"Normalisation = {normalisation}")
            head.append(f"Comment = {comment}")
            return head

        def vector_to_string(
            index: int, vector: np.ndarray | list, decimals: int = 6
        ) -> str:
            """Siemens style vector conversion."""
            return (
                f"Vector[{index}] = ("
                f"{vector[0]: .{decimals}f},"
                f"{vector[1]: .{decimals}f},"
                f"{vector[2]: .{decimals}f})"
                f"\n"
            )

        header = construct_header(filename=diffusion_vector_file, **options)

        with diffusion_vector_file.open("w") as file:
            # write header to file
            for line in header:
                file.write(line + "\n")

            # get diffusion values
            diffusion_vectors = self.get_diffusion_vectors()
            # write values to file
            for row_idx, row in enumerate(diffusion_vectors):
                file.write(vector_to_string(row_idx, row))

        # Neccessary
        # %Berechnung
        # geometrisches
        # Mittel
        # geometric_mean_vector = nthroot(prod(result_vector(:)), R);
        #
        # %Berechnung
        # axialen
        # Diffusivität
        # AD = length_b;
        #
        # %Schreiben in.dvs - Datei
        # fprintf(fileID, 'GeometricMean[%d] = %.6f\n', b, geometric_mean_vector);
        # fprintf(fileID, 'AxialDiffusivity[%d] = %.6f\n', b, AD);

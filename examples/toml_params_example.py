"""
Example script demonstrating how to use TOML parameter files in Pyneapple.

This example shows how to:
1. Load parameters from a TOML file
2. Inspect the loaded parameters
3. Modify parameters
4. Save parameters to a new TOML file
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path if running as a script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from pyneapple.parameters.ivim import IVIMParams
from pyneapple.models import ivim


def main():
    """Run the TOML parameters example."""
    # Path to the sample parameter file
    script_dir = Path(__file__).parent
    sample_params_path = Path(script_dir, "../data/sample_params.toml").resolve()

    print(f"Loading parameters from: {sample_params_path}")

    # Load parameters from TOML file
    try:
        params = IVIMParams(sample_params_path)
        print("\nSuccessfully loaded parameters from TOML file!")

        # Print the loaded parameters
        print("\n--- Loaded Parameters ---")
        print(f"Model: {params.model}")
        print(f"Fit type: {params.fit_type}")
        print(f"Number of components: {params.n_components}")
        print(f"Maximum iterations: {params.max_iter}")

        # Print the boundaries
        print("\n--- Boundaries ---")
        print(f"Scaling: {params.boundaries.scaling}")

        # Access D (diffusion) boundaries
        print("\nDiffusion coefficients:")
        for key, value in params.boundaries.dict.get("D", {}).items():
            print(f"  D_{key}: Initial={value[0]}, Min={value[1]}, Max={value[2]}")

        # Access f (fraction) boundaries
        print("\nFraction values:")
        for key, value in params.boundaries.dict.get("f", {}).items():
            print(f"  f_{key}: Initial={value[0]}, Min={value[1]}, Max={value[2]}")

        # Generate a signal based on the loaded parameters
        if params.b_values is not None and len(params.b_values) > 0:
            print("\n--- Generating Signal Based on Parameters ---")
            b_values = params.b_values.squeeze()

            # Extract initial parameter values for signal generation
            D_slow = params.boundaries.dict["D"]["0"][0]
            D_fast = params.boundaries.dict["D"]["1"][0]
            f_slow = params.boundaries.dict["f"]["0"][0]
            S0 = params.boundaries.dict["S"]["0"][0]

            # Generate signal using the bi-exponential model
            signal = ivim.biexponential_model(
                b_values,
                [S0, f_slow, D_slow, D_fast]
            )

            # Plot the signal
            plt.figure(figsize=(10, 6))
            plt.plot(b_values, signal, 'o-', label='Generated Signal')
            plt.xlabel('b-values (s/mm²)')
            plt.ylabel('Signal')
            plt.title('Signal Generated from TOML Parameters')
            plt.grid(True)
            plt.legend()

            # Save the plot
            output_dir = Path(script_dir, "../output")
            output_dir.mkdir(exist_ok=True)
            plot_path = Path(output_dir, "toml_params_signal.png")
            plt.savefig(plot_path)
            print(f"Signal plot saved to: {plot_path}")

            # Modify and save parameters to a new TOML file
            print("\n--- Modifying Parameters ---")
            params.fit_type = "gpu"  # Change fit type to GPU
            params.max_iter = 200    # Increase max iterations
            params.boundaries.dict["D"]["0"][0] = 0.8  # Change initial value for D_slow

            # Save modified parameters to a new TOML file
            output_params_path = Path(output_dir, "modified_params.toml")
            params.save_to_toml(output_params_path)
            print(f"Modified parameters saved to: {output_params_path}")

        else:
            print("No b-values found in parameters, skipping signal generation.")

    except Exception as e:
        print(f"Error loading or processing parameters: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""Setup script for nash package."""

from setuptools import setup, find_packages

setup(
    name="nash",
    version="0.1.0",
    description="Nash equilibria in quantum games using Matrix Product States",
    packages=find_packages(include=["src", "src.*", "utils", "utils.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "einops",
        "torch",
        "networkx",
        "jaxtyping",
        "tqdm",
        "pyyaml",
    ],
    extras_require={
        "dev": ["pytest", "jupyter", "ipykernel"],
        "mpi": ["mpi4py"],
        "wandb": ["wandb"],
    },
)

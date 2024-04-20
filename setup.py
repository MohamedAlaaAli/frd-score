# coding: utf-8
import os

import setuptools



def read(rel_path):
    base_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_path, rel_path), "r") as f:
        return f.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


if __name__ == "__main__":
    setuptools.setup(
        name="frd",
        version=get_version(os.path.join("src", "frd", "__init__.py")),
        author="Richard Osuala, Preeti Verma",
        description=(
            "Package for calculating Fréchet Radiomics Distance (FRD)"
        ),
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        url="https://github.com/RichardObi/frd",
        project_urls={
            "Bug Tracker": "https://github.com/RichardObi/frd/issues",
        },
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src"),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
        ],
        python_requires=">=3.5",
        entry_points={
            "console_scripts": [
                "frd = frd.frd_score:main",
            ],
        },
        install_requires=[
            "pyradiomics==3.0.1a3", #>=3.1.0", #~=3.0.1a3 #"pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics@releases/tag/v3.1.0",
            "numpy>=1.26.4",
            "Pillow>=10.3.0",
            "scipy>=1.10.0",
            "opencv_contrib_python_headless>=4.8.1.78",
            "SimpleITK>=2.3.1",
            "tqdm>=4.64.1",
        ],
        extras_require={
            "dev": ["flake8", "flake8-bugbear", "flake8-isort", "black==24.3.0", "isort", "nox", "pytest>=8.1.1", "nibabel>=3.2.1", ]
        },
    )
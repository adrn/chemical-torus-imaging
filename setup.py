import os
from setuptools import find_packages, setup

setup(
    name="totoro",
    use_scm_version={
        "write_to": os.path.join("totoro", "version.py"),
        "write_to_template": '__version__ = "{version}"\n',
    },
    description="Orbital Torus Imaging",
    author="adrn",
    author_email="adrianmpw@gmail.com",
    url="https://github.com/adrn/chemical-torus-imaging",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.6",
    # package_data={"totoro": ["data/filename"]},
    # include_package_data=True,
    zip_safe=False,
    install_requires=[
        "numpy",
        "scipy",
        "astropy",
        "matplotlib",
        "astro-gala",
        "galpy",
        "pyia"
    ]
)

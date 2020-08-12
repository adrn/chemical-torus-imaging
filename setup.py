import os
from setuptools import find_packages, setup

setup(
    name="thriftshop",
    use_scm_version={
        "write_to": os.path.join("thriftshop", "version.py"),
        "write_to_template": '__version__ = "{version}"\n',
    },
    description="Check yo tags",
    author="adrn",
    author_email="adrianmpw@gmail.com",
    url="https://github.com/adrn/thriftshop",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.6",
    # package_data={"thriftshop": ["data/filename"]},
    # include_package_data=True,
    zip_safe=False,
    install_requires=[
        "numpy",
        "scipy",
        "astropy",
        "matplotlib",
        "astro-gala"
    ]
)
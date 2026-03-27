from setuptools import setup, find_packages

setup(
    name="npu_support",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    py_modules=["npu_support"],
)

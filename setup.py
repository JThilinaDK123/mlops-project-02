from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name = "MLOps-Project-02",
    version = "1.0",
    author = "Thilina",
    packages = find_packages(),
    install_requires = requirements
)

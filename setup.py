import setuptools

def long_description():
    """Parse long description from readme."""
    with open("README.md", "r") as readme_file:
        return readme_file.read()


setuptools.setup(
    name="amplitf",
    version="0.0.1-alpha2",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/ComPWA/AmpliTF",
    packages=setuptools.find_packages(),
    license="GPLv3 or later",
    python_requires=">=3.6",
    install_requires=[
        "iminuit",
        "numpy",
        "sympy",
        "tensorflow>=2.0"
    ],
    package_data={},
    include_package_data=True,
)

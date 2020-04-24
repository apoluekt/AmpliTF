"""
A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import setuptools


def long_description():
    """Parse long description from readme."""
    with open("README.md", "r") as readme_file:
        return readme_file.read()


setuptools.setup(
    name="amplitf",
    author="Anton Poluektov",
    version='0.0a0',
    long_description=long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/apoluekt/AmpliTF",
    packages=setuptools.find_packages(),
    python_requires='>=3.5',
    install_requires=[
        'iminuit',
        'numpy',
        'sympy',
        'tensorflow>=2.0',
    ],
)

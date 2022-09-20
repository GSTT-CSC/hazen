import setuptools
import os

__version__ = '1.0.2'

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hazen",
    version=__version__,
    url="https://bitbucket.org/gsttmri/hazen",
    author="Shuaib, Haris",
    author_email="mohammad_haris.shuaib@kcl.ac.uk",
    description="An automatic MRI QA tool",
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=install_requires,
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
    entry_points={
        'console_scripts': [
            'hazen = hazenlib:entry_point',
        ],
    },
)

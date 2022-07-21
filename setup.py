import setuptools
import os

VERSION = '0.5.2'

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="hazen",
    version=VERSION,
    url="https://bitbucket.org/gsttmri/hazen",
    author="Shuaib, Haris",
    author_email="mohammad_haris.shuaib@kcl.ac.uk",
    description="An automatic MRI QA tool",
    long_description=open('README.md').read(),
    packages=setuptools.find_packages(),
    install_requires=install_requires,
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

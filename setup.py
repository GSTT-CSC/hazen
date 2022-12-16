import setuptools
import os

__version__ = '1.0.4'

install_requires = ['pydicom==2.2.2',
                    'numpy==1.21.4',
                    'matplotlib==3.5.1',
                    'docopt==0.6.2',
                    'opencv-python-headless==4.6.0.66',
                    'scikit-image==0.19.2',
                    'scipy==1.8.0',
                    'imutils==0.5.3',
                    'colorlog==6.6.0',]

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
    install_requires=install_requires,
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

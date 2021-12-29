import setuptools
import hazenlib

setuptools.setup(
    name="hazen",
    version=hazenlib.__version__,
    url="https://bitbucket.org/gsttmri/hazen",
    author="Shuaib, Haris",
    author_email="mohammad_haris.shuaib@kcl.ac.uk",
    description="An automatic MRI QA tool",
    long_description=open('README.md').read(),
    packages=setuptools.find_packages(),
    install_requires=[],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    entry_points={
        'console_scripts': [
            'hazen = hazenlib:entry_point',
        ],
    },
)

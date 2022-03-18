# -*- coding: utf-8 -*-
from setuptools import setup
import hazenlib

packages = \
['hazenlib']

package_data = \
{'': ['*']}

install_requires = \
['scikit-image==0.19.0',
 'numpy==1.21.4',
 'opencv-python==4.5.4.58',
 'pytest>=6.2',
 'pydicom==1.4.1',
 'matplotlib==3.4.3',
 'scipy==1.7.2',
 'psycopg2-binary==2.8.4',
 'coverage>=6.0.2',
 'imutils==0.5.3',
 'docopt==0.6.2',
 'colorlog==6.6.0'
]

entry_points = \
{
    'console_scripts': [
        'hazen = hazenlib:entry_point',
    ],
}

classifiers = \
[
    'Development Status :: 2 - Pre-Alpha',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9'
]

setup_kwargs = {
    'name': "hazen",
    'version': hazenlib.__version__,
    'description': 'Quality assurance framework for Magnetic Resonance Imaging',
    'long_description': open('README-PyPI.md').read(),
    'long_description_content_type': 'text/markdown',
    'author': 'GSTT-CSC',
    'author_email': 'mohammad_haris.shuaib@kcl.ac.uk',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/GSTT-CSC/hazen',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<4.0',
    'entry_points': entry_points
}

setup(**setup_kwargs)

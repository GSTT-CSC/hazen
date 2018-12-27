import setuptools

setuptools.setup(
    name="hazen",
    version="dev-0.1.0",
    url="https://bitbucket.org/gsttmri/hazen",
    author="Shuaib, Mohammad Haris",
    author_email="mohammad_haris.shuaib@kcl.ac.uk",
    description="An automatic MRI QA tool",
    long_description=open('README.rst').read(),
    packages=setuptools.find_packages(),
    install_requires=[],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)

from setuptools import setup, find_packages

# Package info
PACKAGE_NAME = "seanalysis"
SHORT_DESCRIPTION = 'Tools for Search Engine Similarity Analysis'

PACKAGES_ROOT = '.'
PACKAGES = find_packages(PACKAGES_ROOT)

# Package meta
CLASSIFIERS = []

# Package requirements
INSTALL_REQUIRES = []

EXTRAS_REQUIRES = {}

TESTS_REQUIRES = ['mock', 'nose']


setup(
    name=PACKAGE_NAME,
    version="0.1",
    description=SHORT_DESCRIPTION,
    classifiers=CLASSIFIERS,
    packages=PACKAGES,
    package_dir={'': PACKAGES_ROOT},
    include_package_data=True,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRES,
    tests_require=TESTS_REQUIRES,
    entry_points={
        'console_scripts': [
            'seanlz = seanalysis.cli.cmd:main'
        ]
    }
)

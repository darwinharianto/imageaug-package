from setuptools import setup

setup(
    name='imageaug',
    version='0.0.1',
    description='Currently used as a wrapper for imgaug by aleju',
    url='git@github.com:darwinharianto/imageaug-package.git',
    author='Darwin Harianto',
    author_email='hariantodarwin@gmail.com',
    license='unlicense',
    packages=['imageaug'],
    install_requires=[
        'opencv-python>=4.1.1.26',
        'numpy>=1.17.2',
        'Shapely>=1.6.4.post2',
        'matplotlib>=3.1.1',
        'Pillow>=6.1.0',
        'pycocotools>=2.0.0',
        'pylint>=2.4.2',
        'labelme>=3.16.7',
        'PyYAML>=5.1.2',
        'pyqt5>=5.14.1',
        'pyclay-common_utils @ https://github.com/cm107/common_utils/archive/master.zip',
        'pyclay-logger @ https://github.com/cm107/logger/archive/master.zip',
        'pyclay-annotation_utils @ https://github.com/cm107/annotation_utils/archive/master.zip'
    ],
    zip_safe=False
)

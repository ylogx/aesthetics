from distutils.core import setup
from setuptools import find_packages

def get_version():
    return '0.0.2'

def get_requirements():
    with open('requirements.txt', 'rU') as fhan:
        requires = [line.strip() for line in fhan.readlines()]
    return requires

def get_long_description():
    try:
        import pypandoc
        long_description = pypandoc.convert('README.md', 'rst')
    except (IOError, ImportError):
        with open('README.txt') as fhan:
            long_description = fhan.read()
    return long_description


add_keywords = dict(
    entry_points={
        'console_scripts': ['aesthetics = aesthetics.cli:main'],
    }, )

setup(
    name='Aesthetics',
    description='Image Aesthetics Toolkit',
    version=get_version(),
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    license='GPLv3+',
    author='Shubham Chaudhary',
    author_email='me@shubhamchaudhary.in',
    url='https://github.com/shubhamchaudhary/aesthetics',
    long_description=get_long_description(),
    install_requires=get_requirements(),
    **add_keywords)

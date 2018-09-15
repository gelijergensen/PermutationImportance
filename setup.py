"""Setup file for PermutationImportance"""

from setuptools import setup

PACKAGE_NAMES = ['permutation_importance']
KEYWORDS = [
    'variable importance', 'model evaluation']
SHORT_DESCRIPTION = (
    'Important variables determined through permutation selection')
LONG_DESCRIPTION = (
    'Provides an efficient method to compute variable importance through the '
    'permutation of input variables. Uses multithreading and supports both '
    'Windows and Unix systems and Python 2 and 3.')

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2.7']

PACKAGE_REQUIREMENTS = [
    'numpy', 'ctypes', 'multiprocessing']

if __name__ == '__main__':
    setup(name='PermutationImportance', version='1.0',
          description=SHORT_DESCRIPTION,
          author='G. Eli Jergensen', author_email='gelijergensen@ou.edu',
          long_description=LONG_DESCRIPTION, license='MIT',
          url='https://github.com/gelijergensen/Variable-Importance',
          packages=PACKAGE_NAMES, scripts=[], keywords=KEYWORDS,
          classifiers=CLASSIFIERS, include_package_data=True, zip_safe=False,
          install_requires=PACKAGE_REQUIREMENTS)

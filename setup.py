"""Setup file for PermutationImportance"""

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

PACKAGE_NAMES = ['permutation_importance']
KEYWORDS = [
    'variable importance', 'model evaluation']
SHORT_DESCRIPTION = (
    'Important variables determined through permutation selection')

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering :: Information Analysis']

PACKAGE_REQUIREMENTS = ['numpy']

if __name__ == '__main__':
    setup(name='PermutationImportance', version='1.1.0.1',
          description=SHORT_DESCRIPTION,
          author='G. Eli Jergensen', author_email='gelijergensen@ou.edu',
          long_description=long_description,
          long_description_content_type="text/markdown",
          license='MIT',
          url='https://github.com/gelijergensen/PermutationImportance',
          packages=PACKAGE_NAMES, scripts=[], keywords=KEYWORDS,
          classifiers=CLASSIFIERS, include_package_data=True, zip_safe=False,
          install_requires=PACKAGE_REQUIREMENTS)

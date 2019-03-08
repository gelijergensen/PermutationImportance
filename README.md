# <span class="permutationimportancetitle">PermutationImportance</span>

[![Build Status](https://travis-ci.com/gelijergensen/PermutationImportance.svg?branch=master)](https://travis-ci.com/gelijergensen/PermutationImportance)
[![Documentation Status](https://readthedocs.org/projects/permutationimportance/badge/?version=latest)](https://permutationimportance.readthedocs.io/en/latest/?badge=latest)

![PermutationImportance Logo](docs/images/favicon.png)

<link rel="stylesheet" href="docs/_static/css/stylesheet.css">
Welcome to the <span 
class="permutationimportancetitle">PermutationImportance</span> library!

<span class="permutationimportancetitle">PermutationImportance</span> is a
Python package for Python 2.7 and 3.5+ which provides several methods for
computing data-based predictor importance. The methods implemented are
model-agnostic and can be used for any machine learning model in many stages of
development. The complete documentation can be found at our
[Read The Docs](https://permutationimportance.readthedocs.io/en/latest/).

## Version History

- 1.2.1.5: Added documentation and examples and ensured compatibility with
  Python 3.5+
- 1.2.1.4: Original scores are now also bootstrapped to match the other results
- 1.2.1.3: Corrected an issue with multithreading deadlock when returned scores
  were too large
- 1.2.1.1: Provided object to assist in constructing scoring strategies
  - Also added two new strategies with bootstrapping support
- 1.2.1.0: Metrics can now accept kwargs and support bootstrapping
- 1.2.0.0: Added support for Sequential Selection and completely revised backend
  for proper abstraction and extension
  - Return object now keeps track of `(context, result)` pairs
  - `abstract_variable_importance` enables implementation of custom variable
    importance methods
  - Backend is now correctly multithreaded (when specified) and is
    OS-independent
- 1.1.0.0: Revised return object of Permutation Importance to support easy
  retrieval of Breiman- and Lakshmanan-style importances
- 1.0.0.0: Published with `pip` support!

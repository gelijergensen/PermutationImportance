# PermutationImportance

Provides an efficient method to compute variable importance through the
permutation of input variables. Uses multithreading and supports both Windows
and Unix systems and Python 2 and 3.

This repository provides the stand-alone functionality of a permutation-based
method of computing variable importance in an arbitrary model. Importance for a
given variable is computed in accordance with Lakshmanan et al. (2015)'s
paper[1]. Variables which, when their values are permuted, cause the worst
resulting score are considered most important. This implementation provides the
functionality for an arbitrary method for computing the "worst" score and for
using an arbitrary scoring metric. The most common case of this is to chose the
variable which most negatively impacts the accuracy of the model.

Functionality is provided not only for returning the most important variable
(along with the raw scores for each variable) but also for returning the
sequential importance of variables. To do this, the most important variable is
determined and then it is left permuted while the next most important variable
is determined. In the extreme case, this is continued until the sequential
ordering of all variables is determined, but this can be terminated at an
earlier level by choice.

<sup>1</sup>Lakshmanan, V., C. Karstens, J. Krause, K. Elmore, A. Ryzhkov, and
S. Berkseth, 2015: Which Polarimetric Variables Are Important for
Weather/No-Weather Discrimination?. J. Atmos. Oceanic Technol., 32, 1209–1223,
https://doi.org/10.1175/JTECH-D-13-00205.1

## Setup

PermutationImportance is now available on pip, so you can simply install with

```bash
pip install PermutationImportance
```

and import the desired method with

```python
from permutation_importance.variable_importance import permutation_selection_importance
```

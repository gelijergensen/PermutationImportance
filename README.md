# Variable-Importance

This repository provides the stand-alone functionality of a permutation-based
method of computing variable importance in an arbitrary model.

Importance for a given variable is computed in accordance with Lakshmanan et al.
(2015)'s paper[1]. Variables which, when their values are permuted, cause the
worst resulting score are considered most important. This implementation
provides the functionality for an arbitrary method for computing the ``worst''
score and for using an arbitrary scoring metric. The most common case of this is
to chose the variable which most negatively impacts the accuracy of the model.

Functionality is provided not only for returning the most important variable
(along with the raw scores for each variable) but also for returning the
sequential importance of variables. To do this, the most important variable is
determined and then it is left permuted while the next most important variable
is determined. In the extreme case, this is continued until the sequential
ordering of all variables is determined, but this can be terminated at an
earlier level by choice.

The implementation here works in both Python 2.7 and 3 and on both Windows and
Unix systems (although the sharing of variables between threads is not possible
on Windows) and is parallelized to greatly speed up computation time.

<sup>1</sup>Lakshmanan, V., C. Karstens, J. Krause, K. Elmore, A. Ryzhkov, and
S. Berkseth, 2015: Which Polarimetric Variables Are Important for
Weather/No-Weather Discrimination?. J. Atmos. Oceanic Technol., 32, 1209–1223,
https://doi.org/10.1175/JTECH-D-13-00205.1

## Setup

TODO: Use setuptools so that you can just have people clone the repo and then
call `python setup.py install`

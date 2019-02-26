.. title:: Levels of Abstraction

.. _levels_of_abstraction:

Levels of Abstraction
=====================

Model-Based
-----------

Typically, variable importance is computed with respect to a 
particular model. In this case, the function for scoring is in fact either a
performance metric or an error or loss function. All functions of this type are
prefixed with ``sklearn_`` because they are designed for use primarily with 
scikit-learn models.

Method-Specific
---------------

In some cases, variable importance is computed either over 
the dataset itself (rather than a model) or for a model which is incompatible
with scikit-learn. Here, the function for scoring must be given in terms of the
training data and the scoring data. If you wish to score using a model which is
not compatible with scikit-learn, you may still find utility in the tools 
provided in PermutationImportance.sklearn_api. All functions of this type are 
named specifically for the method they employ

Method-Agnostic
---------------

There are other data-based methods for computing variable
importance beyond the ones implemented here. If you wish to design your own
variable importance, you can still take advantage of the generalized algorithm
for computing data-based variable importances as well as the multithreaded 
functionality implemented in "abstract_variable_importance". In order to use
this function, you will need to design your own strategy for providing the 
datasets to be used at each iteration. Please see 
PermutationImportance.abstract_runner.abstract_variable_importance and
PermutationImportance.selection_strategies for more information.

TODO fix the links here
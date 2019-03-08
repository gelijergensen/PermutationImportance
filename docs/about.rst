.. role:: permutationimportancetitle

.. title:: About PermutationImportance

*********************************************************
About :permutationimportancetitle:`PermutationImportance`
*********************************************************

.. _pi_image:
.. image:: ./images/favicon.png  
   :align: right
   :scale: 200%



Welcome to the :permutationimportancetitle:`PermutationImportance` library! 

:permutationimportancetitle:`PermutationImportance` is a data-science library which provides several data-based methods for computing the importance of predictors in a machine learning model. Recently, machine learning has achieved breakthrough in a number of fields, but despite its observed successes and its wide adoption in many domains, machine learning is often criticized as being a "black box". Often, users do not understand how a machine learning model makes its predictions and are hesitant to rely on a device which seems to pull its predictions out of the air. Predictor importance evaluation is one technique used to help allieviate the interpretability problem.

Evaluation of predictor importance, like many other methods for model interpretation, is useful in several phases of machine learning model development. Initially, it can be used to aid with debugging by identifying predictors upon which the model is either relying too heavily or predictors which the model is ignoring entirely. Later, as the model is going into production, predictor importance provides a evaluation tool which highlights the skills of the model and hints at its strengths and shortcomings. Finally, if the model surpasses human skill, predictor importance can provide insight into the workings of the model, allowing us to learn from its decisions. Whether as a diagnostic, evaluative, or didactic tool, predictor importance can help guide and support machine learning model development.

There are several methods implemented in this library, such as permutation importance, the namesake for this package. All methods are model-agnostic and will work for all machine learning models. All require data to be used for training and/or scoring, a function for scoring given the data, and a function for converting the scores to relative rankings of the predictors. For more information on a particular method, please see the documentation for that method.

************
Installation
************

:permutationimportancetitle:`PermutationImportance` can be easily installed using pip::

  pip install -U PermutationImportance

You can ensure that you have the lastest version by checking

.. doctest::

  >>> import PermutationImportance
  >>> print(PermutationImportance.__version__)
  1.2.1.5


.. _levels_of_abstraction:

*********************
Levels of Abstraction
*********************

Model-Based
===========

Typically, variable importance is computed with respect to a 
particular model. In this case, the function for scoring is in fact either a
performance metric or an error or loss function. All functions of this type are
prefixed with ``sklearn_`` because they are designed for use primarily with 
scikit-learn models.

- :func:`PermutationImportance.permutation_importance.sklearn_permutation_importance`
- :func:`PermutationImportance.sequential_selection.sklearn_sequential_forward_selection`
- :func:`PermutationImportance.sequential_selection.sklearn_sequential_backward_selection`

Method-Specific
===============

In some cases, variable importance is computed either over 
the dataset itself (rather than a model) or for a model which is incompatible
with scikit-learn. Here, the function for scoring must be given in terms of the
training data and the scoring data. If you wish to score using a model which is
not compatible with scikit-learn, you may still find utility in the tools 
provided in :mod:`PermutationImportance.sklearn_api`. All functions of this type are 
named specifically for the method they employ.

- :func:`PermutationImportance.permutation_importance.permutation_importance`
- :func:`PermutationImportance.sequential_selection.sequential_forward_selection`
- :func:`PermutationImportance.sequential_selection.sequential_backward_selection`

Method-Agnostic
===============

There are other data-based methods for computing variable
importance beyond the ones implemented here. If you wish to design your own
variable importance, you can still take advantage of the generalized algorithm
for computing data-based variable importances as well as the multithreaded 
functionality implemented in :func:`PermutationImportance.abstract_runner.abstract_variable_importance`. In order to use
this function, you will need to design your own strategy for providing the 
datasets to be used at each iteration. For more information, please see 

- :mod:`PermutationImportance.abstract_runner`
- :mod:`PermutationImportance.selection_strategies` 

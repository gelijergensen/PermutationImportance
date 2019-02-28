.. title:: Custom Methods

Custom Methods
==============

While we provide a number of data-based methods out of the box, you may find that you wish to implement a data-based predictor importance method which we have not provided. For convenience, we provide tools that may assist in the process of implementing those methods.

Firstly, we provide the function :ref:`abstract_variable_importance <abstract_variable_importance>`, which encapsulates the general process of performing a data-based predictor importance method and additionally provides automatic hooks into both the single- and multi-process backends. So long as the method which you wish to implement follows the general structure of "scoring" given ``(training_data, scoring_data)`` tuples to evaluate importance for each predictor in succession, you should be able to use the :ref:`abstract_variable_importance <abstract_variable_importance>` function directly by only providing a valid :ref:`selection_strategy <selection_strategy>`. For more on this process, so below. Even if your desired method does not match this pattern, you may still find utility in the two backends. 
TODO link


Abstract Variable Importance
----------------------------

The :ref:`abstract_variable_importance <abstract_variable_importance>` function handles the generalized process for computing predictor importance. The algorithm itself consists of a double-``for`` loop, the first of which loops once for each of the predictors, the second of which loops over the list of triples ``(predictor, training_data, scoring_data)`` returned by the given :ref:`selection_strategy <selection_strategy>`. This allows for the majority of the implementation details to be left to the :ref:`selection_strategy <selection_strategy>`.

-----

.. _abstract_variable_importance:
.. autofunction:: PermutationImportance.abstract_runner.abstract_variable_importance
   :noindex:


Selection Strategy
------------------

The :ref:`selection_strategy <selection_strategy>` is the most important part of a predictor importance method, as it essentially defines the method. In this code, a ``SelectionStrategy`` is an object which is initialized with the original ``training_data`` and ``scoring_data`` datasets passed to the predictor importance method, the total number of variables, and the current variables which are considered important. It must act as a generator which yields tuples of ``(variable, training_data_subset, scoring_data_subset)``. This can be thought of as yielding the information to test the importance of this ``variable`` by using the ``training_data_subset`` and ``scoring_data_subset``.

For convenience, we provide the base ``SelectionStrategy`` object, which should be extended to make a new method. Each object should have a static ``name`` property (for diagnostics) and should override the ``generate_all_datasets`` or ``generate_datasets`` method. As many methods test precisely the predictors which are not yet considered important, the default implementation of ``generate_all_datasets`` calls ``generate_datasets`` once for each currently unimportant predictor. Please see the implementation of the base ``SelectionStrategy`` object, as well as the other classes in PermutationImportance.selection_strategies for more details.
TODO Link

-----

.. _selection_strategy:
.. autoclass:: PermutationImportance.selection_strategies.SelectionStrategy
   :members: __init__, generate_all_datasets, generate_datasets
   :noindex:



.. title:: Metrics

.. _sklearn_metrics: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

.. _metrics:

Metrics
=======

These are metric functions which can be used to score model predictions 
against the true values. They are designed to be used either as a component of
a ``scoring_fn`` of the method-specific methods or stand-alone 
as the ``evaluation_fn`` of a model-based method. In addition to these metrics, all of the metrics and loss functions provided by
`Scikit-Learn <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_ should also work.

-----

.. autofunction:: PermutationImportance.metrics.gerrity_score

-----

.. autofunction:: PermutationImportance.metrics.peirce_skill_score

-----

.. autofunction:: PermutationImportance.metrics.heidke_skill_score


.. title:: ImportanceResult

.. _importance_result:

ImportanceResult
================

The ``ImportanceResult`` is an object which keeps track of the full context and
scoring determined by a variable importance method. Because the variable 
importance methods iteratively determine the next most important variable, this
yields a sequence of pairs of "contexts" (i.e. the previous ranks/scores of 
variables) and "results" (i.e. the current ranks/scores of variables). This
object keeps track of those pairs and additionally provides methods for the easy
retrieve of both the results with empty context (singlepass, Breiman) and the
most complete context (multipass, Lakshmanan). Further, it enables iteration 
over the ``(context, results)`` pairs and for indexing into the list of pairs.

-----

.. autoclass:: PermutationImportance.ImportanceResult
   :members: __init__, retrieve_singlepass, retrieve_multipass
   :noindex:
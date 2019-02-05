# PermutationImportance

[![Build Status](https://travis-ci.com/gelijergensen/PermutationImportance.svg?branch=master)](https://travis-ci.com/gelijergensen/PermutationImportance)

### Update (05 Feb. 2019):

The documentation is not yet up-to-date for the current version (1.2.0.0). We
expect to have that complete within a few days, but for the time being, if you
wish to see how to use the most recent version, please look at the file
`test/test_integration.py` for example uses or email me with questions.

#

Provides an efficient method to compute variable importance through the
permutation of input variables. Uses multithreading and supports both Windows
and Unix systems and Python 2 and 3.

This repository provides the stand-alone functionality of a permutation-based
method of computing variable importance in an arbitrary model. Importance for a
given variable is computed in accordance with either Breiman (2001)'s paper[1]
or Lakshmanan et al. (2015)'s paper[2]. Variables which, when their values are
permuted, cause the worst resulting score are considered most important. This
implementation provides the functionality for an arbitrary method for computing
the "worst" score and for using an arbitrary scoring metric. The most common
case of this is to chose the variable which most negatively impacts the accuracy
of the model.

Breiman offers an O(n) algorithm for this, which orders the variables according
to those whose scoring is most adversely affected by permutation. In this
repository, this is referred to as **variable importance**. Lakshmanan offers an
O(n^2) algorithm which determines the next most important variable given a
previous listing of important variables. This ensures that if multiple variables
are highly correlated, then only one will appear as important. In this
repository, this is referred to as **variable rank**.

Functionality is provided not only for returning the variable importance and
ranks, but also for the scores which determined those rankings, which may be
useful for visualization.

<sup>1</sup>Breiman, Leo. "Random forests." Machine learning 45.1 (2001): 5-32.
https://doi.org/10.1023/A:1010933404324

<sup>2</sup>Lakshmanan, V., C. Karstens, J. Krause, K. Elmore, A. Ryzhkov, and
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

## Example Usage

For these examples, I will assume a particular model (`FakeModel` below), but
any python object with a `predict` method can be used.

```python
class FakeModel(object):
    """Computes a particular linear combination of the input variables of a dataset"""

    def __init__(self, cutoff, *coefs):
        """Store the coeficients to multiply the input variables by"""
        self.cutoff = cutoff
        self.coefs = coefs

    def predict(self, input_data):
        return np.array([sum([self.coefs[i]*data[i] for i in range(len(data))]) > self.cutoff for data in input_data])

model = FakeModel(20, 3, 2, 0, 1)
```

You will also need the testing inputs and outputs, both of which should be
`numpy.ndarrays`. Similarly, a `classes` object containing the possible outputs
of the model is required for some scoring function. For my example, I am using

```python
# Binary classification with 4 input variables
# x is most important (scaled by a factor of 3)
# y is next most important (scaled by a factor of 2)
# z is not important at all
# w is slightly important (scaled by 1)
def population_def(x, y, z, w): return int(3*x + 2*y + w > 20)
    fake_model_input = np.random.randint(0, 10, size=(1000, 4))
    fake_model_output = [population_def(*data_point)
                         for data_point in fake_model_input]
    classes = [0, 1]
```

Lastly, for the most simple usage of the model, you also require the number of
permutations to perform. Generally, this should be at least 30 to ensure better
statistics. Here is the simplest call:

```python
results = permutation_selection_importance(model, classes, fake_model_input, fake_model_output, npermute=30)
# Returns an object which contains the importances, ranks, and scores which determined those
importances, original_score, importances_scores = results.get_variable_importances_and_scores() # O(n) algorithm (Breiman's)
ranks, original_score, rank_scores = results.get_variable_ranks_and_scores() # O(n^2) algorithm (Lakshmanan's)
```

### Options

#### nimportant_variables

By default, the algorithm will execute the entire `O(n`<sup>`2`</sup>`)` version
of the algorithm (Lakshmanan's), but if you only want the first `m` entries of
`ranks`, you can specify `nimportant_variables=m` in the function call. If you
only care about Breiman's version of the algorithm, set
`nimportant_variables=1`. e.g.

```python
results = permutation_selection_importance(model, classes, fake_model_input, fake_model_output, npermute=30, nimportant_variables=1)
importances, original_score, importances_scores = results.get_variable_importances_and_scores() # still intact
ranks, original_score, rank_scores = results.get_variable_ranks_and_scores()  # only one element long... not very useful
```

#### subsample

By default, the algorithm will use the entire data for each permutation
iteration, but you will typically want to only use a smaller fraction (say
half). For this, you can specify `subsample=0.5`

```python
results = permutation_selection_importance(model, classes, fake_model_input, fake_model_output, npermute=30, subsample=0.5)
```

#### score_fn

Any arbitrary callable can be used for scoring, so long as it has the form
`(new_predictions, truths, classes) -> number`. Normally, this defaults to
accuracy. Some example scorers are provided in the `scorers.py` file. For
example, to use Peirce Skill Score

```python
from permutation_importance.scorers import peirce_skill_scorer
results = permutation_selection_importance(model, classes, fake_model_input, fake_model_output, npermute=30, score_fn=peirce_skill_scorer)
```

#### optimization

A different strategy can be used for determining which variable is the most
important, given a list of scores by specifying
`optimization=<string or callable>`. By default, the score which is lowest after
permuting indicates that variable was most important (`"minimize"`), but
(`"maximize"`) is also supported. Further, an arbitrary callable can be used, so
long as it has the form `(list_of_scores) -> index`. This is particularly useful
for scoring function like bias, where a bias of 1 is best and a bias near 0 or
+infinity indicates the model is doing poorly. Assuming some sort of "error"
function where a lower score is better, the example usage is

```python
error_fn # some arbitrary error function where a lower score is better
results = permutation_selection_importance(model, classes, fake_model_input, fake_model_output, npermute=30, score_fn=error_fn, optimization='maximize')
```

#### njobs and share_vars

The implementation can be multithreaded by specifying a number of jobs
(`njobs`). Additionally, for optimization, as many variables as possible can be
shared between threads `share_vars=True`. Sadly, this feature only works on Unix
systems, so the parameter `share_vars` will be ignored on Windows systems. To
multithread the algorithm for a Unix machine with 8 cores, call

```python
results = permutation_selection_importance(model, classes, fake_model_input, fake_model_output, npermute=30, njobs=8, share_vars=True)
```

This same command will also run for a Windows machine with 8 cores, as the
`share_vars` parameter will be safely ignored.

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
    classes = 20
```

Lastly, for the most simple usage of the model, you also require the number of
permutations to perform. Generally, this should be at least 30 to ensure better
statistics. Here is the simplest call:

```python
results = permutation_selection_importance(model, classes, fake_model_input, fake_model_output, npermute=30)
# Returns a list of triples which is as long as the number of input variables
len(results)
# 4
(important_variable_indices_ordered, score_before_permuting, all_scores_after_permuting) = results[i]
```

For the triple in the `k`th element of `results`,
`important_variable_indices_ordered` contains a list of the indices of the `k`
variables which are most important together. `results[0][0]` is a list
containing only the index of the most important individual variable.
`results[1][0]` contains the index of the most important variable and then the
index of the variable which is most important after the first variable has been
removed. `results[2][0]` contains those same two indices followed by the index
of the variable which is most important after those two variables have been
removed, etc. So for the example above

```python
results[0][0]
# [0]
results[1][0]
# [0, 1]
results[2][0]
# [0, 1, 3]
results[3][0]
# [0, 1, 3, 2]
```

In this manner, `results[k][0]` contains the ordering of the "ranks" of the
`k`+1 most important variables (Section 2.e of the paper).

`score_before_permuting` is the score after permuting all variables of a "rank"
before the (`k`+1)th variable, and `all_scores_after_permuting` is a list of the
scores for all of the variables, with `None` for the scores of the variables of
"rank" lower than `k`+1. (These may be handy for plotting the results.)

There are two typical ways to determine the relative importances of each
variable:

1. Use the "ranks" of all the variables: `results[-1][0]`. This takes
   `O(n`<sup>`2`</sup>`)` time
2. Use the ordering of the scores after each variable is permuted only once:
   `np.argsort(results[0][2])[::-1]`. This takes `O(n)` time

```python
results = permutation_selection_importance(model, classes, fake_model_input, fake_model_output, npermute=30)

rank_importances = results[-1][0]
# Reverse the order here because the variable with the worst score is the most important
permutation_importances = np.argsort(results[0][2])[::-1]
```

### Options

#### nimportant_variables

By default, the algorithm will execute the `O(n`<sup>`2`</sup>`)` version of the
algorithm (1. above), but if you only want the first `m` entries of `results`,
you can specify `nimportant_variables=m` in the function call. To execute the
`O(n)` version of the algorithm (2. above), set `nimportant_variables=1`. e.g.

```python
results = permutation_selection_importance(model, classes, fake_model_input, fake_model_output, npermute=30, nimportant_variables=1)
importances = np.argsort(results[0][2])[::-1]
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

"""This class details the return type of the variable importance"""


class PermutationImportanceResult(object):
    """This object contains the result of the permutation results. In general, one will only need to use the two
    convenience functions which return the variable importances and scores or the variable ranks and scores.

    However, should more information be required, this object also keeps track of the following:
    model (model importance was computed for)
    optimization (function used to compute the "least optimal" resulting score)
    score_fn (function used to score the predictions of the model)
    nimportant_variables (number of iterations the variable importance was computer for)
    all_scores (all of the scores for each iteration (a list of lists, where None values were not computed))
    complete_results (the raw output of the recursive variable importance algorithm. Should be entirely redundant)
    """

    def __init__(self, model, optimization, score_fn, nimportant_variables, complete_results):
        """Bundles the complete_results returned by the recursive variable importances code into a much neater object

        : param model: model for which the permutation importance was computed
        : param optimization: function which determines which resulting score is least optimal
        : param score_fn: function for computing the score of the predictions
        : param ntimportant_variables: number of variables to compute permutation ranks for
        : param complete_results: result of the recursive variable importance code
        """
        self.model = model
        self.optimization = optimization
        self.score_fn = score_fn
        self.nimportant_variables = nimportant_variables
        self._convert_results_to_attributes(complete_results)
        self.complete_results = complete_results

    def _convert_results_to_attributes(self, complete_results):
        """Use the complete_results object to compute the variable importances and scores and attach those to self

        : param complete_results: result of the recursive variable importance code
        """
        # last iteration's ordered list of important variables
        variable_rank_indices = complete_results[-1][0]

        self.all_scores = [result[-1] for result in complete_results]
        self.original_score = complete_results[0][1]
        self.scores_for_permutation_importance, self.scores_for_permutation_rank = self._compute_scores_for_permutation_importance_and_rank(
            complete_results)

        num_variables = len(self.scores_for_permutation_importance)

        self.permutation_importances = self._compute_variable_importances(
            self.scores_for_permutation_importance)
        self.permutation_ranks = self._compute_variable_ranks(
            num_variables, variable_rank_indices)

    def _compute_scores_for_permutation_importance_and_rank(self, complete_results):
        """Compute the scores that determined the permutation importance and permutation rank

        : param complete_results: result of the recursive variable importance code
        : returns: scores_for_permutation_importance, scores_for_permutation_rank
        """
        scores_for_permutation_importance = complete_results[0][-1]
        scores_for_permutation_rank = [
            complete_results[i+1][1] for i in range(self.nimportant_variables - 1)]
        # The last score needs to be computed using the optimization function (sorry!)
        final_scores = complete_results[-1][-1]
        valid_scores = list()
        for score in final_scores:
            if score is not None:
                valid_scores.append(score)
        scores_for_permutation_rank.append(
            valid_scores[self.optimization(valid_scores)])
        return scores_for_permutation_importance, scores_for_permutation_rank

    def _compute_variable_ranks(self, num_variables, variable_rank_indices):
        """Converts an ordered list of indices corresponding to important variables to the ranks

        : param num_variables: total number of variables input to the model
        : param variable_rank_indices: ordered list of important variable indices
        : returns: a list of the ranks of each variable, where None indicates that it has no computed rank
        """
        variable_ranks = [None for _ in range(num_variables)]
        for rank, idx in enumerate(variable_rank_indices):
            variable_ranks[idx] = rank
        return variable_ranks

    def _compute_variable_importances(self, scores_for_permutation_importance):
        """Determines the ordering for permutation importance using only the scores for permutation importance

        : param scores_for_permutation_importance: scores for each variable after only that column was permuted
        : returns: a list of the ordering of the variable importances
        """
        # The only way we can determine the correct ordering is to iteratively apply the optimization function
        num_vars = len(scores_for_permutation_importance)
        ordering = list()
        remaining_vars = [i for i in range(num_vars)]
        scores_to_test = scores_for_permutation_importance[:]   # ensure a copy
        for i in range(num_vars):
            next_best = self.optimization(scores_to_test)
            ordering.append(remaining_vars.pop(next_best))
            scores_to_test.pop(
                next_best)  # Also remove score from listing

        # convert the ordering into the importances
        variable_importances = [None for _ in range(num_vars)]
        for importance, idx in enumerate(ordering):
            variable_importances[idx] = importance
        return variable_importances

    def get_variable_importances_and_scores(self):
        """Convenience method which returns the permutation importances and the scores which determined the importances.
        This is the result of the O(n) way of doing this(Breiman)"""
        return self.permutation_importances, self.original_score, self.scores_for_permutation_importance

    def get_variable_ranks_and_scores(self):
        """Convenience method which returns the permutation ranks and the scores which determined the ranks. This is the
        result of the O(n ^ 2) way of doing this(Lakshmanan)"""
        return self.permutation_ranks, self.original_score, self.scores_for_permutation_rank

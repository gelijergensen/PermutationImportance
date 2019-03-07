"""This is an example of how to plot. Feel free to either use this code as is
or to make modifications to it as you see fit."""

import matplotlib
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


# Set up the font sizes for matplotlib
FONT_SIZE = 14
BIG_FONT_SIZE = FONT_SIZE + 2
LARGE_FONT_SIZE = FONT_SIZE + 4
HUGE_FONT_SIZE = FONT_SIZE + 6
SMALL_FONT_SIZE = FONT_SIZE - 2
TINY_FONT_SIZE = FONT_SIZE - 4
TEENSIE_FONT_SIZE = FONT_SIZE - 6
font_sizes = {
    'teensie': TEENSIE_FONT_SIZE,
    'tiny': TINY_FONT_SIZE,
    'small': SMALL_FONT_SIZE,
    'normal': FONT_SIZE,
    'big': BIG_FONT_SIZE,
    'large': LARGE_FONT_SIZE,
    'huge': HUGE_FONT_SIZE,
}
plt.rc('font', size=FONT_SIZE)
plt.rc('axes', titlesize=FONT_SIZE)
plt.rc('axes', labelsize=FONT_SIZE)
plt.rc('xtick', labelsize=TINY_FONT_SIZE)
plt.rc('ytick', labelsize=TINY_FONT_SIZE)
plt.rc('legend', fontsize=FONT_SIZE)
plt.rc('figure', titlesize=BIG_FONT_SIZE)


def plot_variable_importance(importance_obj, filename, multipass=True, relative=False, num_vars_to_plot=10, diagnostics=0):
    """Plots any variable importance method for a particular estimator

    :param importance_obj: ImportanceResult object returned by PermutationImportance
    :param filename: string to place the file into (including directory and '.png')
    :param multipass: whether to plot multipass or singlepass results. Default to True
    :param relative: whether to plot the absolute value of the results or the results relative to the original. Defaults
        to plotting the absolute results
    :param num_vars_to_plot: number of top variables to actually plot (cause otherwise it won't fit)
    :param diagnostics: 0 for no printouts, 1 for all printouts, 2 for some printouts. defaults to 0
    """

    rankings = importance_obj.retrieve_multipass(
    ) if multipass else importance_obj.retrieve_singlepass()

    original_score = importance_obj.original_score

    try:
        len(original_score)
    except:
        bootstrapped = False
    else:
        bootstrapped = True

    if bootstrapped:
        original_score_mean = np.mean(original_score)
    else:
        original_score_mean = original_score

    # Sort by increasing rank
    sorted_var_names = list(rankings.keys())
    sorted_var_names.sort(key=lambda k: rankings[k][0])
    sorted_var_names = sorted_var_names[:min(num_vars_to_plot, len(rankings))]
    scores = [rankings[var][1] for var in sorted_var_names]

    colors_to_plot = [variable_to_color(var) for var in [
        "Original Score", ] + sorted_var_names]
    variable_names_to_plot = [" {}".format(
        var) for var in convert_vars_to_readable(["Original Score", ] + sorted_var_names)]

    if bootstrapped:
        if relative:
            scores_to_plot = np.array([original_score_mean, ] + [np.mean(score)
                                                                 for score in scores]) / original_score_mean
        else:
            scores_to_plot = np.array(
                [original_score_mean, ] + [np.mean(score) for score in scores])
        ci = np.array([np.abs(np.mean(score) - np.percentile(score, [0.025, 0.975]))
                       for score in np.r_[[original_score, ], scores]]).transpose()
    else:
        if relative:
            scores_to_plot = np.array(
                [original_score_mean, ] + scores) / original_score_mean
        else:
            scores_to_plot = np.array(
                [original_score_mean, ] + scores)
        ci = np.array([[0, 0]
                       for score in np.r_[[original_score, ], scores]]).transpose()

    metric = "Score"
    if importance_obj.method == "Permutation Importance":
        method = "%s Permutation Importance" % (
            "Multipass" if multipass else "Singlepass")
    else:
        method = importance_obj.method
    title = "%s\n%s" % (metric, method)

    # Actually make plot
    fig = plt.figure(figsize=(4, 3))
    if bootstrapped:
        plt.barh(np.arange(len(scores_to_plot)),
                 scores_to_plot, linewidth=1, edgecolor='black', color=colors_to_plot, xerr=ci, capsize=4, ecolor='grey', error_kw=dict(alpha=0.4))
    else:
        plt.barh(np.arange(len(scores_to_plot)),
                 scores_to_plot, linewidth=1, edgecolor='black', color=colors_to_plot)

    # Put the variable names _into_ the plot
    for i in range(len(variable_names_to_plot)):
        plt.text(0, i, variable_names_to_plot[i],
                 va="center", ha="left", size=font_sizes['teensie'])
    if relative:
        plt.axvline(1, linestyle=':', color='grey')
        plt.text(1, len(variable_names_to_plot) / 2, "original score = %0.3f" % original_score_mean,
                 va='center', ha='left', size=font_sizes['teensie'], rotation=270)
        plt.xlabel("Percent of Original Score")
        plt.xlim([0, 1.2])
    else:
        plt.axvline(original_score_mean, linestyle=':', color='grey')
        plt.text(original_score_mean, len(variable_names_to_plot) / 2, "original score",
                 va='center', ha='left', size=font_sizes['teensie'], rotation=270)
        plt.xlabel("Score")

    plt.ylabel("Predictor Permuted")
    plt.title(title)
    plt.yticks([])
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # make the horizontal plot go with the highest value at the top
    plt.gca().invert_yaxis()

    print("Saving file to %s" % filename)
    plt.savefig(filename, dpi=300, bbox_inches="tight")


# You can fill this in by using a dictionary with {var_name: legible_name}
def convert_vars_to_readable(variables_list):
    """Substitutes out variable names for human-readable ones
    :param variables_list: a list of variable names
    :returns: a copy of the list with human-readable names
    """
    human_readable_list = list()
    for var in variables_list:
        if False:  # var in VARIABLE_NAMES_DICTONARY:
            pass  # human_readable_list.append(VARIABLE_NAMES_DICTONARY[var])
        else:
            human_readable_list.append(var)
    return human_readable_list


# This could easily be expanded with a dictionary
def variable_to_color(var):
    return "lightblue"

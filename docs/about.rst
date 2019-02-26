.. title:: About

About
-----

Welcome to the PermutationImportance library! 

PermutationImportance is a data-science library which provides several data-based methods for computing the importance of predictors in a machine learning model. Recently, machine learning has achieved breakthrough in a number of fields, but despite its observed successes and its wide adoption in many domains, machine learning is often criticized as being a "black box". Often, users do not understand how a machine learning model makes its predictions and are hesitant to rely on a device which seems to pull its predictions out of the air. Predictor importance evaluation is one technique used to help allieviate the interpretability problem.

Evaluation of predictor importance, like many other methods for model interpretation, is useful in several phases of machine learning model development. Initially, it can be used to aid with debugging by identifying predictors upon which the model is either relying too heavily or predictors which the model is ignoring entirely. Later, as the model is going into production, predictor importance provides a evaluation tool which highlights the skills of the model and hints at its strengths and shortcomings. Finally, if the model surpasses human skill, predictor importance can provide insight into the workings of the model, allowing us to learn from its decisions. Whether as a diagnostic, evaluative, or didactic tool, predictor importance can help guide and support machine learning model development.

There are several methods implemented in this library, such as permutation importance, the namesake for this package. All methods are model-agnostic and will work for all machine learning models. All require data to be used for training and/or scoring, a function for scoring given the data, and a function for converting the scores to relative rankings of the predictors. For more information on a particular method, please see the documentation for that particular method.
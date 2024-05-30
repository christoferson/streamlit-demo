

opt_help_temperature = """
Temperature - Affects the shape of the probability distribution for the predicted output and influences the likelihood of the model selecting lower-probability outputs.

Choose a lower value to influence the model to select higher-probability outputs.

Choose a higher value to influence the model to select lower-probability outputs.

In technical terms, the temperature modulates the probability mass function for the next token. A lower temperature steepens the function and leads to more deterministic responses, and a higher temperature flattens the function and leads to more random responses.
"""



opt_help_top_k = """
Top K - The number of most-likely candidates that the model considers for the next token.

Choose a lower value to decrease the size of the pool and limit the options to more likely outputs.

Choose a higher value to increase the size of the pool and allow the model to consider less likely outputs.

For example, if you choose a value of 50 for Top K, the model selects from 50 of the most probable tokens that could be next in the sequence.
"""

opt_help_top_p = """
Top P – The percentage of most-likely candidates that the model considers for the next token.

Choose a lower value to decrease the size of the pool and limit the options to more likely outputs.

Choose a higher value to increase the size of the pool and allow the model to consider less likely outputs.

In technical terms, the model computes the cumulative probability distribution for the set of responses and considers only the top P% of the distribution.

For example, if you choose a value of 0.8 for Top P, the model selects from the top 80% of the probability distribution of tokens that could be next in the sequence.
"""
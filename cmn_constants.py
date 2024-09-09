

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


css_sidebar = """
<style>
[data-testid="stSidebarContent"] {
    color: white;
    background-color: none;
}
[data-testid="stSidebarNav"] {
    color: white;
    background-color: none;
}
[data-testid="stSidebarNavItems"] {
    color: white;
    background-color: none;
    scrollbar-color: lightgray lightblue;
    overflow-y: scroll;
}

[data-testid="stSidebarNavSeparator"] {
    color: white;
    background-color: none;
    
}

</style>
"""

css_sidebar_2 = """
    <style>
    div.stCodeBlock {
        background-color: transparent;
    }
    div.stCodeBlock > pre {
        background-color: transparent;
    }
    code.language-wiki {
        font-size: 16px;
        font-family: "Source Sans Pro", sans-serif;
        line-height: 1.6;
        max-width: 100%;
        display: inline-block;
        word-wrap: break-word;
        word-break: break-all;
        white-space: pre-line;
        overflow-wrap: anywhere;
        color: blue;
        background-color: transparent;
    }
    code.language-wiki > span {
        font-size: 16px;;
        font-family: "Source Sans Pro", sans-serif;
        line-height: 1.6;
        max-width: 100%;
        display: inline-block;
        word-wrap: break-word;
        word-break: break-all;
        white-space: pre-line;
        overflow-wrap: anywhere;
        color: blue;
        background-color: transparent;
    }

    code.language-markdown {
        font-size: 16px;
        font-family: "Source Sans Pro", sans-serif;
        line-height: 1.6;
        max-width: 100%;
        display: inline-block;
        word-wrap: break-word;
        word-break: break-all;
        white-space: pre-line;
        overflow-wrap: anywhere;
        color: orange;
        background-color: transparent;
    }

    </style>
    """


css_btn_primary = """
<style>
[data-testid="stSidebarContent"] {
    color: white;
    background-color: none;
}
[data-testid="stSidebarNav"] {
    color: white;
    background-color: none;
}
[data-testid="stSidebarNavItems"] {
    color: white;
    background-color: none;
    scrollbar-color: lightgray lightblue;
    overflow-y: scroll;
}

[data-testid="stSidebarNavSeparator"] {
    color: white;
    background-color: none;
    
}

</style>
"""


css_button_primary = """
    <style>
    button[kind="primary"] {
        background: none!important;
        border: none;
        padding: 0!important;
        margin: 0;
        color: black !important;
        text-decoration: none;
        cursor: pointer;
        border: none !important;
    }
    button[kind="primary"]:hover {
        text-decoration: none;
        color: black !important;
    }
    button[kind="primary"]:focus {
        outline: none !important;
        box-shadow: none !important;
        color: black !important;
    }
    </style>
    """
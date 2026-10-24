"""Grounded reasoning traces for Super Mario Bros, after nuReasoning (arXiv 2605.31572).

Pipeline (see README.md): key frames where the player's behaviour changes -> exact facts from
game RAM -> the human's decision as a meta-action -> simulated counterfactual alternatives ->
a one-sentence trace written by a VLM with hindsight, checked against all of the above.
"""

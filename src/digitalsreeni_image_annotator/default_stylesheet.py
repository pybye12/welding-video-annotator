"""Light theme stylesheet.

Generated from the shared tokens in ``theme.py`` so the light and dark
sheets cannot drift apart. Change colours or metrics there, not here.
"""

from .theme import LIGHT_TOKENS, build_stylesheet

default_stylesheet = build_stylesheet(LIGHT_TOKENS)

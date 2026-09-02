"""Dark theme stylesheet.

Generated from the shared tokens in ``theme.py`` so the light and dark
sheets cannot drift apart. Change colours or metrics there, not here.
"""

from .theme import DARK_TOKENS, build_stylesheet

soft_dark_stylesheet = build_stylesheet(DARK_TOKENS)

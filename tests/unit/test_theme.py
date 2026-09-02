import re

import pytest

from digitalsreeni_image_annotator.default_stylesheet import default_stylesheet
from digitalsreeni_image_annotator.soft_dark_stylesheet import soft_dark_stylesheet
from digitalsreeni_image_annotator.theme import (
    DARK_TOKENS,
    LIGHT_TOKENS,
    build_stylesheet,
    tokens_for,
)


def test_both_themes_define_the_same_tokens():
    """A token added to one theme only would render an unstyled widget."""
    assert set(LIGHT_TOKENS) == set(DARK_TOKENS)


def test_every_token_is_a_colour_or_the_theme_name():
    for tokens in (LIGHT_TOKENS, DARK_TOKENS):
        for key, value in tokens.items():
            if key == "name":
                continue
            assert re.fullmatch(r"#[0-9A-Fa-f]{6}", value), f"{key}={value!r}"


def test_stylesheets_are_generated_from_the_token_tables():
    assert default_stylesheet == build_stylesheet(LIGHT_TOKENS)
    assert soft_dark_stylesheet == build_stylesheet(DARK_TOKENS)


def test_no_unsubstituted_placeholders_remain():
    """A typo'd token name would render as a literal {placeholder}."""
    for sheet in (default_stylesheet, soft_dark_stylesheet):
        # QSS braces are structural; the only single braces left after
        # rendering would be an unsubstituted field.
        assert not re.search(r"\{[a-z_]+\}", sheet)
        assert "None" not in sheet


def test_a_token_missing_from_one_theme_fails_loudly():
    """The whole point of generating both sheets from one template."""
    incomplete = dict(LIGHT_TOKENS)
    del incomplete["accent"]

    with pytest.raises(KeyError):
        build_stylesheet(incomplete)


def test_both_sheets_carry_every_selector_the_template_defines():
    """Guards against a rule that only renders under one theme."""

    def selectors(sheet):
        return {
            line.strip().rstrip("{").strip()
            for line in sheet.splitlines()
            if line.strip().endswith("{") or line.strip().endswith(",")
        }

    light, dark = selectors(default_stylesheet), selectors(soft_dark_stylesheet)
    assert light == dark
    # Not vacuous: the app relies on these being styled in both themes.
    for required in (
        "QPushButton[buttonRole=\"primary\"]",
        "QWidget#controlPanel,",
        "QStatusBar",
        "QProgressBar#frameProgress::chunk",
        "QLabel.help-text",
    ):
        assert required in light, required


def test_tokens_for_selects_by_mode():
    assert tokens_for(True) is DARK_TOKENS
    assert tokens_for(False) is LIGHT_TOKENS


def test_status_colours_are_distinct_from_the_accent():
    """Labeled-frame green must not read as 'selected'."""
    for tokens in (LIGHT_TOKENS, DARK_TOKENS):
        assert tokens["success"] != tokens["accent"]
        assert tokens["danger"] != tokens["accent"]

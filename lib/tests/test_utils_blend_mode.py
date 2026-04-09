import pytest
from lib.utils_blend_mode import OverlapBlendMode, get_blend_factor


class TestGetBlendFactor:
    @pytest.mark.parametrize("percent,expected", [
        (0.0, 1.0),
        (0.2, 0.8),
        (0.4, 0.6),
        (0.6, 0.4),
        (0.8, 0.2),
        (1.0, 0.0),
    ])
    def test_linear_blend_factor(self, percent, expected):
        assert get_blend_factor(OverlapBlendMode.LINEAR_BLEND, percent) == expected

    @pytest.mark.parametrize("percent,expected", [
        (0.0, 1.0),
        (0.2, 0.896),
        (0.4, 0.648),
        (0.6, 0.352),
        (0.8, 0.104),
        (1.0, 0.0),
    ])
    def test_ease_in_out_factor(self, percent, expected):
        assert get_blend_factor(OverlapBlendMode.EASE_IN_OUT, percent) == expected

    @pytest.mark.parametrize("percent,expected", [
        (0.0, 1.0),
        (0.2, 0.0),
        (0.4, 0.0),
        (0.6, 0.0),
        (0.8, 0.0),
        (1.0, 0.0),
    ])
    def test_newer_factor(self, percent, expected):
        assert get_blend_factor(OverlapBlendMode.NEWER, percent) == expected

    @pytest.mark.parametrize("percent,expected", [
        (0.0, 1.0),
        (0.2, 1.0),
        (0.4, 1.0),
        (0.6, 1.0),
        (0.8, 1.0),
        (1.0, 0.0),
    ])
    def test_older_factor(self, percent, expected):
        assert get_blend_factor(OverlapBlendMode.OLDER, percent) == expected

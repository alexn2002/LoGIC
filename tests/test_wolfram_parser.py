from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devices import validate_covariance
from pipeline import _identify_txt_style, _parse_wolfram_text


class WolframParserTests(unittest.TestCase):
    def test_wolfram_rationals_and_scientific_powers_parse(self) -> None:
        parsed = _parse_wolfram_text(
            "{{1/5, {9.229922294035418*^-7, 3 I}}, {-2/5, {1/10 I, 0}}}"
        )

        self.assertEqual(parsed[0][0], 0.2)
        self.assertEqual(parsed[0][1][0], 9.229922294035418e-7)
        self.assertEqual(parsed[0][1][1], 3j)
        self.assertEqual(parsed[1][0], -0.4)
        self.assertEqual(parsed[1][1][0], 0.1j)

    def test_wolfram_precision_marks_constants_and_numeric_functions_parse(self) -> None:
        parsed = _parse_wolfram_text(
            "{1.234567890123456`20.*^-5, Pi, E, -Pi/2, Sqrt[2], 1/Sqrt[2], 2^3, Power[4, 1/2]}"
        )

        self.assertAlmostEqual(parsed[0], 1.234567890123456e-5)
        self.assertAlmostEqual(parsed[1], np.pi)
        self.assertAlmostEqual(parsed[2], np.e)
        self.assertAlmostEqual(parsed[3], -np.pi / 2)
        self.assertAlmostEqual(parsed[4], np.sqrt(2))
        self.assertAlmostEqual(parsed[5], 1 / np.sqrt(2))
        self.assertEqual(parsed[6], 8)
        self.assertEqual(parsed[7], 2.0)

    def test_wolfram_complex_forms_parse(self) -> None:
        parsed = _parse_wolfram_text(
            "{1 + 2 I, 1 - 2 I, -I, 3.5`30. I, Complex[1/2, -3*^-2], Exp[I Pi]}"
        )

        self.assertEqual(parsed[0], 1 + 2j)
        self.assertEqual(parsed[1], 1 - 2j)
        self.assertEqual(parsed[2], -1j)
        self.assertEqual(parsed[3], 3.5j)
        self.assertEqual(parsed[4], 0.5 - 0.03j)
        self.assertAlmostEqual(parsed[5].real, -1.0)
        self.assertAlmostEqual(parsed[5].imag, 0.0)

    def test_wolfram_function_syntax_does_not_look_like_matlab(self) -> None:
        self.assertEqual(_identify_txt_style("{0, {{Complex[1, 2]}}}"), "wolfram_style")

    def test_wolfram_comments_and_named_characters_parse(self) -> None:
        parsed = _parse_wolfram_text(r"{(* comment *) \[Pi], \[ExponentialE], 2 \[ImaginaryI]}")

        self.assertAlmostEqual(parsed[0], np.pi)
        self.assertAlmostEqual(parsed[1], np.e)
        self.assertEqual(parsed[2], 2j)

    def test_wolfram_zero_denominators_fail_clearly(self) -> None:
        for payload in ("{1/0}", "{Rational[1, 0]}"):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "zero denominator"):
                    _parse_wolfram_text(payload)

    def test_unknown_wolfram_symbol_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported Wolfram"):
            _parse_wolfram_text("{NotANumericSymbol}")

    def test_covariance_validation_allows_text_rounding_noise(self) -> None:
        d = np.zeros(2)

        validate_covariance((0.5 - 2.1e-10) * np.eye(2), d)

        with self.assertRaises(ValueError):
            validate_covariance((0.5 - 2.0e-8) * np.eye(2), d)


if __name__ == "__main__":
    unittest.main()

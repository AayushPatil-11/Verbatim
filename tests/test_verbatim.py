import unittest

from verbatim import (
    Claim,
    UNSUPPORTED_QUOTE_REASON,
    verify_and_render,
    verify_claims,
)


class VerbatimTests(unittest.TestCase):
    def test_verify_claims_marks_supported_and_unsupported(self):
        document = "Tenant must provide 30 days written notice before move-out."
        claims = [
            Claim("Notice period is 30 days", "30 days written notice"),
            Claim("Notice period is 60 days", "60 days written notice"),
        ]

        verified = verify_claims(document, claims)

        self.assertTrue(verified[0].supported)
        self.assertIsNone(verified[0].failure_reason)
        self.assertFalse(verified[1].supported)
        self.assertEqual(verified[1].failure_reason, UNSUPPORTED_QUOTE_REASON)

    def test_verify_and_render_formats_output(self):
        document = "Pets are allowed with a refundable deposit of $250."
        claims = [
            Claim("Pets are allowed", "Pets are allowed"),
            Claim("Deposit is $500", "$500"),
        ]

        rendered = verify_and_render(document, claims)

        self.assertIn("<mark>Pets are allowed</mark>", rendered["highlighted_source"])
        self.assertIn('- ❌ ~~Deposit is $500~~', rendered["claims"])
        self.assertIn('attempted quote: "$500"', rendered["claims"])
        self.assertIn(UNSUPPORTED_QUOTE_REASON, rendered["claims"])


if __name__ == "__main__":
    unittest.main()

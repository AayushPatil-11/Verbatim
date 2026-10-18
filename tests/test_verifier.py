import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.verifier import DocIndex, clean_quote, trust_score  # noqa: E402

DOC = (
    "4. SECURITY DEPOSIT. The Landlord shall refund the Security Deposit, without\n"
    "interest, within thirty (30) days after the Tenant vacates the Premises.\n"
    "5. The Landlord may not enter without 24 hours’ notice."
)


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.idx = DocIndex(DOC)

    def test_exact_match_ignores_line_breaks_and_case(self):
        m = self.idx.find("the security deposit, without interest, WITHIN thirty (30) days")
        self.assertEqual(m.status, "exact")
        # offsets point at the original text
        self.assertTrue(DOC[m.start:m.end].lower().startswith("the security deposit"))
        self.assertTrue(DOC[m.start:m.end].endswith("days"))

    def test_curly_apostrophe_matches_straight(self):
        m = self.idx.find("enter without 24 hours' notice")
        self.assertEqual(m.status, "exact")

    def test_invented_quote_is_not_found(self):
        m = self.idx.find("The Landlord shall refund the deposit within seven (7) days of vacating.")
        self.assertFalse(m.found)

    def test_fuzzy_match_tolerates_small_noise(self):
        m = self.idx.find("shall refund the Security Deposit, without interest within thirty (30) day after the Tenant vacates")
        self.assertEqual(m.status, "fuzzy")
        self.assertGreaterEqual(m.score, 90)

    def test_fuzzy_rejects_changed_number(self):
        m = self.idx.find("shall refund the Security Deposit, without interest, within thirty (45) days after the Tenant vacates")
        self.assertFalse(m.found)
        self.assertIn("Numbers", m.reason)

    def test_fuzzy_rejects_dropped_negation(self):
        m = self.idx.find("The Landlord may enter without 24 hours notice")
        self.assertFalse(m.found)

    def test_empty_and_short_quotes_rejected(self):
        self.assertFalse(self.idx.find("").found)
        self.assertFalse(self.idx.find("the Tenant").found)
        self.assertFalse(self.idx.find(None).found)

    def test_clean_quote_strips_wrappers_and_ellipses(self):
        self.assertEqual(clean_quote('"... within thirty (30) days ..."'), "within thirty (30) days")

    def test_trust_score(self):
        claims = [
            {"status": "supported", "match": "exact"},
            {"status": "supported", "match": "fuzzy"},
            {"status": "retracted", "match": None},
            {"status": "unsupported", "match": "exact"},
        ]
        t = trust_score(claims)
        self.assertEqual((t["supported"], t["unsupported"], t["retracted"], t["total"]), (2, 1, 1, 4))
        self.assertEqual(t["score"], round(100 * 1.9 / 4))
        self.assertEqual(trust_score([])["score"], 0)


if __name__ == "__main__":
    unittest.main()

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402
from app.mock import DEMO_QUESTION, SAMPLE_PATH, MockLLM  # noqa: E402
from app.pipeline import PipelineError, parse_json, run  # noqa: E402

DOC = SAMPLE_PATH.read_text()


class PipelineTests(unittest.TestCase):
    def test_demo_run_covers_every_status(self):
        r = run(DOC, DEMO_QUESTION, MockLLM())
        statuses = [c["status"] for c in r["claims"]]
        self.assertEqual(statuses, ["supported", "supported", "supported", "retracted", "unsupported"])
        self.assertEqual(r["claims"][2]["match"], "fuzzy")
        retracted = r["claims"][3]
        self.assertTrue(retracted["retried"])
        self.assertTrue(retracted["quote"])            # original cited quote is kept
        self.assertIn("does not appear", retracted["reason"])
        self.assertNotIn("7 days", r["verified_answer"])
        self.assertEqual(r["trust"]["total"], 5)

    def test_offsets_point_at_real_text(self):
        r = run(DOC, DEMO_QUESTION, MockLLM())
        c = r["claims"][0]
        self.assertIn("thirty (30) days", r["document_text"][c["start"]:c["end"]])

    def test_without_entailment_layer(self):
        r = run(DOC, DEMO_QUESTION, MockLLM(), entailment=False)
        self.assertEqual(r["claims"][4]["status"], "supported")
        self.assertFalse(r["entailment_enabled"])

    def test_model_that_invents_everything_is_fully_retracted(self):
        class Liar:
            mode = "live"
            def complete(self, system, user, stage=""):
                if stage == "claims":
                    return json.dumps({"claims": [{"claim": "Rent is Rs. 10.", "quote": "Rent is only ten rupees per month always."}]}), "x", "y"
                if stage == "retry":
                    return json.dumps({"quote": None}), "x", "y"
                return "draft", "x", "y"
        r = run(DOC, "rent?", Liar())
        self.assertEqual(r["claims"][0]["status"], "retracted")
        self.assertEqual(r["trust"]["score"], 0)

    def test_bad_model_json_raises_clean_error(self):
        class Broken:
            mode = "live"
            def complete(self, system, user, stage=""):
                return "I refuse to answer in JSON", "x", "y"
        with self.assertRaises(PipelineError):
            run(DOC, "rent?", Broken())

    def test_parse_json_handles_fences(self):
        self.assertEqual(parse_json('Sure!\n```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json('noise {"a": 2} noise'), {"a": 2})

    def test_mock_supports_only_sample(self):
        self.assertTrue(MockLLM.supports(DOC, DEMO_QUESTION))
        self.assertFalse(MockLLM.supports("some other document text here", DEMO_QUESTION))


class _FakeAPI(BaseHTTPRequestHandler):
    calls = {"groq": 0, "or": 0}

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        which = "groq" if self.path.startswith("/groq") else "or"
        _FakeAPI.calls[which] += 1
        if which == "groq":
            self.send_response(429)
            self.end_headers()
            return
        body = json.dumps({"choices": [{"message": {"content": "hello from openrouter"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


class LLMClientTests(unittest.TestCase):
    def test_falls_back_to_openrouter_after_rate_limit(self):
        server = HTTPServer(("127.0.0.1", 0), _FakeAPI)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        saved = [dict(p) for p in llm.PROVIDERS]
        llm.PROVIDERS[0]["base"] = base + "/groq"
        llm.PROVIDERS[1]["base"] = base + "/or"
        os.environ["GROQ_API_KEY"] = "k1"
        os.environ["OPENROUTER_API_KEY"] = "k2"
        try:
            text, provider, model = llm.LLMClient(timeout=5).complete("s", "u")
        finally:
            server.shutdown()
            for p, s in zip(llm.PROVIDERS, saved):
                p.update(s)
            os.environ.pop("GROQ_API_KEY"); os.environ.pop("OPENROUTER_API_KEY")
        self.assertEqual((text, provider), ("hello from openrouter", "openrouter"))
        self.assertEqual(_FakeAPI.calls["groq"], 2)   # one retry on 429, then fallback

    def test_no_keys_raises(self):
        for k in ("GROQ_API_KEY", "OPENROUTER_API_KEY"):
            os.environ.pop(k, None)
        with self.assertRaises(llm.LLMError):
            llm.LLMClient().complete("s", "u")


if __name__ == "__main__":
    unittest.main()

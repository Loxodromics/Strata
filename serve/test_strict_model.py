"""serve/test_strict_model.py - the config's "strict_model" refuses requests that name another model.

Without it every name is served by the loaded model, so a client switched to a model that is not loaded gets
answers under the wrong name and sizes its prompts for that model's context.

Runs against the mock-engine server (no GPU, no network).

    python -m unittest serve.test_strict_model -v
"""
from __future__ import annotations

import http.client
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPLY = "<think>\nhm\n</think>\n\nhello from the mock"
LOADED, ALIAS, OTHER = "loaded-model", "other-name-of-it", "some-other-model"
MESSAGES = [{"role": "user", "content": "hi"}]
BODIES = {
    "/v1/chat/completions": {"messages": MESSAGES, "max_tokens": 32},
    "/v1/messages": {"messages": MESSAGES, "max_tokens": 32},
    "/v1/messages/count_tokens": {"messages": MESSAGES},
    "/v1/responses": {"input": "hi", "max_output_tokens": 32},
}


class StrictModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, REPLY, max_context=4096)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), model_name=LOADED)
        cls.svc.set_aliases([ALIAS])
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.svc.strict_model = True
        self.engine.last_prompt = []

    def post(self, path, body):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            connection.request("POST", path, body=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, response.getheader("Content-Type"), response.read()
        finally:
            connection.close()

    def test_off_by_default_serves_any_name(self):
        tok = ByteTokenizer()
        self.assertFalse(Service(self.engine, tok, self.svc.template).strict_model)
        self.svc.strict_model = False
        for path, body in BODIES.items():
            status, _, _ = self.post(path, {**body, "model": OTHER})
            self.assertEqual(status, 200, path)

    def test_the_loaded_name_and_an_alias_are_served(self):
        for path, body in BODIES.items():
            for name in (LOADED, ALIAS):
                status, _, _ = self.post(path, {**body, "model": name})
                self.assertEqual(status, 200, (path, name))

    def test_a_request_naming_no_model_is_served(self):
        for name in ({}, {"model": None}, {"model": ""}):
            status, _, _ = self.post("/v1/chat/completions", {**BODIES["/v1/chat/completions"], **name})
            self.assertEqual(status, 200, name)

    def test_another_name_is_refused_before_the_engine(self):
        for path, body in BODIES.items():
            for stream in (False, True):
                status, content_type, raw = self.post(path, {**body, "model": OTHER, "stream": stream})
                self.assertEqual(status, 404, (path, stream))
                self.assertIn("application/json", content_type)
                message = json.loads(raw)["error"]["message"]
                self.assertIn(OTHER, message)
                self.assertIn(LOADED, message)
        self.assertEqual(self.engine.last_prompt, [])

    def test_error_shape_follows_the_api(self):
        def refused(path):
            return json.loads(self.post(path, {**BODIES[path], "model": OTHER})[2])

        self.assertEqual(refused("/v1/chat/completions")["error"]["code"], "model_not_found")
        self.assertEqual(refused("/v1/responses")["error"]["code"], "model_not_found")
        for path in ("/v1/messages", "/v1/messages/count_tokens"):
            body = refused(path)
            self.assertEqual((body["type"], body["error"]["type"]), ("error", "not_found_error"), path)

    def test_a_name_that_is_not_text_is_refused(self):
        for name in (7, ["x"], {"id": LOADED}):
            status, _, _ = self.post("/v1/chat/completions", {**BODIES["/v1/chat/completions"], "model": name})
            self.assertEqual(status, 404, name)

    def test_refusals_do_not_use_up_the_server(self):
        for _ in range(20):
            self.post("/v1/chat/completions", {**BODIES["/v1/chat/completions"], "model": OTHER})
        status, _, _ = self.post("/v1/chat/completions", {**BODIES["/v1/chat/completions"], "model": LOADED})
        self.assertEqual(status, 200)

    def test_a_long_name_is_cut_in_the_message(self):
        message = self.svc.model_refusal({"model": "x" * 10_000})
        self.assertLess(len(message), 400)


if __name__ == "__main__":
    unittest.main()

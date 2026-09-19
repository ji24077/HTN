"""Cross-runtime device authentication and tamper-proof output acceptance."""

import base64
import hashlib
import json
import shutil
import subprocess
import unittest
from uuid import uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from orchestrator.shared.dwp import (
    assertion_identity,
    parse_frame,
    validate_public_key,
    verify_assertion,
    verify_result,
)


def b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


class DeviceCryptoTests(unittest.TestCase):
    def setUp(self):
        self.key = Ed25519PrivateKey.generate()
        self.public = base64.b64encode(
            self.key.public_key().public_bytes(
                serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
            )
        ).decode()
        self.worker = str(uuid4())
        self.claims = {
            "iss": self.worker,
            "aud": "dwp-control",
            "iat": 1000,
            "exp": 1120,
            "jti": str(uuid4()),
        }

    def token(self, claims=None, header=None):
        signed = ".".join(
            b64(json.dumps(value).encode())
            for value in (
                header or {"alg": "EdDSA", "typ": "JWT"},
                self.claims if claims is None else claims,
            )
        )
        return signed + "." + b64(self.key.sign(signed.encode()))

    def result(self, raw_output='{"score":1e-7,"label":"caf\u00e9"}'):
        payload = {
            "taskId": "task-1",
            "attempt": 2,
            "leaseId": "lease-1",
            "output": json.loads(raw_output),
            "outputHash": hashlib.sha256(raw_output.encode()).hexdigest(),
            "startedAt": "2026-09-19T08:00:00.001Z",
            "finishedAt": "2026-09-19T08:00:01.001Z",
        }
        signed = (
            f"dwp-attest/v1\ntask-1\n2\n{self.worker}\n{payload['outputHash']}\n"
            f"{payload['startedAt']}\n{payload['finishedAt']}\n"
        )
        payload["signature"] = b64(self.key.sign(signed.encode()))
        return payload, raw_output

    def verify(self, payload, raw_output, **kwargs):
        values = {"task_id": "task-1", "attempt": 2, "worker_id": self.worker}
        values.update(kwargs)
        return verify_result(payload, raw_output, self.public, **values)

    def test_assertion_valid_signature_identity_and_public_key(self):
        token = self.token()
        self.assertEqual(assertion_identity(token), self.worker)
        self.assertEqual(verify_assertion(token, self.public, now=1001), self.claims)
        self.assertEqual(validate_public_key(self.public), self.public)
        with self.assertRaises(ValueError):
            validate_public_key("not-a-key")

    def test_assertion_rejects_malformed_time_and_routing_claims(self):
        overrides = [
            {"iat": None},
            {"exp": None},
            {"iat": "1000"},
            {"exp": True},
            {"exp": 1120.5},
            {"exp": 1000},
            {"exp": 999},
            {"exp": 1121},
            {"iat": 1200, "exp": 1320},
            {"aud": "other"},
            {"iss": "not-a-uuid"},
            {"iss": self.worker.upper()},
            {"jti": "nonce"},
            {"jti": None},
            {"exp": 2**53},
        ]
        for override in overrides:
            with self.subTest(override=override), self.assertRaises(ValueError):
                verify_assertion(self.token(self.claims | override), self.public, now=1001)
        for header in ({"alg": "none", "typ": "JWT"}, {"alg": "HS256", "typ": "JWT"}):
            with self.assertRaises(ValueError):
                verify_assertion(self.token(header=header), self.public, now=1001)

    def test_assertion_signature_tampering_rejected(self):
        parts = self.token().split(".")
        parts[1] = b64(json.dumps(self.claims | {"jti": str(uuid4())}).encode())
        with self.assertRaises(ValueError):
            verify_assertion(".".join(parts), self.public, now=1001)
        with self.assertRaises(ValueError):
            verify_assertion(self.token(), self.public, now=1120)

    def test_frame_preserves_original_output_numbers_order_and_unicode(self):
        raw_output = '{"z":1e-7,"a":1.0,"nested":{"output":["caf\u00e9",-0]}}'
        raw = (
            '{ "payload" : {"other":{"output":0}, "output" : ' + raw_output + '},"t":"task.result"}'
        )
        frame, original = parse_frame(raw.encode())
        self.assertEqual(original, raw_output)
        self.assertEqual(frame["payload"]["output"]["z"], 1e-7)
        self.assertIsNone(parse_frame('{"payload":{}}')[1])

    def test_frame_rejects_duplicate_keys_nonfinite_and_oversize_input(self):
        for raw in (
            '{"payload":{},"payload":{}}',
            '{"payload":{"output":{"a":1,"a":2}}}',
            '{"payload":{"output":1,"\\u006futput":2}}',
            '{"payload":{"output":NaN}}',
            '{"payload":{"output":Infinity}}',
            '{"payload":{"output":1e999}}',
            "[]",
            b"\xff",
            '{"payload":{"output":"' + "x" * (128 * 1024) + '"}}',
        ):
            with self.subTest(raw=str(raw)[:70]), self.assertRaises(ValueError):
                parse_frame(raw)

    def test_result_evidence_preserves_verifiable_original_bytes(self):
        payload, raw_output = self.result()
        proof = self.verify(payload, raw_output)
        self.assertEqual(proof["rawOutput"], raw_output)
        self.assertEqual(proof["publicKey"], self.public)
        self.assertEqual(proof["hostId"], self.worker)

    def test_result_rejects_unsigned_output_or_assignment_changes(self):
        payload, raw_output = self.result()
        changes = [
            {"output": {"score": 9}},
            {"outputHash": "0" * 64},
            {"attempt": 3},
            {"attempt": True},
            {"taskId": "different"},
            {"startedAt": "2026-09-19T08:00:00.000Z"},
            {"finishedAt": "2026-09-19T08:00:00.000Z"},
            {"signature": b64(bytes(64))},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.verify(payload | change, raw_output)
        with self.assertRaises(ValueError):
            self.verify(payload, raw_output, worker_id=str(uuid4()))
        with self.assertRaises(ValueError):
            self.verify(payload, raw_output.replace("1e-7", "0.0000001"))

    @unittest.skipUnless(shutil.which("node"), "Node is needed for JS signature interoperability")
    def test_real_node_assertion_and_floating_point_result_verify_in_python(self):
        program = r"""
const c = require('node:crypto');
const key = c.generateKeyPairSync('ed25519');
const host = c.randomUUID();
const claims = {iss:host,aud:'dwp-control',iat:1000,exp:1120,jti:c.randomUUID()};
const b64 = v => Buffer.from(JSON.stringify(v)).toString('base64url');
const input = b64({alg:'EdDSA',typ:'JWT'})+'.'+b64(claims);
const token = input+'.'+c.sign(null,Buffer.from(input),key.privateKey).toString('base64url');
const output = {small:1e-7,large:1e21,negative:-0,label:'caf\u00e9',scores:[0.1,1.2345678901234567]};
const p = {taskId:'task-1',attempt:2,output,
  outputHash:c.createHash('sha256').update(JSON.stringify(output)).digest('hex'),
  startedAt:'2026-09-19T08:00:00.001Z',finishedAt:'2026-09-19T08:00:01.001Z'};
const signed = `dwp-attest/v1\n${p.taskId}\n${p.attempt}\n${host}\n${p.outputHash}\n${p.startedAt}\n${p.finishedAt}\n`;
p.signature = c.sign(null,Buffer.from(signed),key.privateKey).toString('base64url');
console.log(JSON.stringify({host,token,publicKey:key.publicKey.export({format:'der',type:'spki'}).toString('base64'),
  frame:JSON.stringify({t:'task.result',payload:p})}));
"""
        run = subprocess.run(
            [shutil.which("node"), "-e", program],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        data = json.loads(run.stdout)
        self.assertEqual(
            verify_assertion(data["token"], data["publicKey"], now=1001)["iss"], data["host"]
        )
        frame, raw = parse_frame(data["frame"])
        proof = verify_result(
            frame["payload"],
            raw,
            data["publicKey"],
            task_id="task-1",
            attempt=2,
            worker_id=data["host"],
        )
        self.assertIn('"large":1e+21', proof["rawOutput"])


if __name__ == "__main__":
    unittest.main()

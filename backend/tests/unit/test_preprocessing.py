import base64
import io
import stat
import unittest
import zipfile

from orchestrator.preprocessing.artifacts import unpack
from orchestrator.preprocessing.comparison import compare, equivalent
from orchestrator.preprocessing.models import UploadFile


class UploadTests(unittest.TestCase):
    def zip(self, entries):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, value in entries:
                archive.writestr(name, value)
        return [
            UploadFile(name="project.zip", content=base64.b64encode(buffer.getvalue()).decode())
        ]

    def test_zip_retains_project_structure_and_original_bytes(self):
        files = unpack(
            self.zip([("project/main.py", b"print('hello')\n"), ("project/data.csv", b"a,b\n1,2")])
        )
        self.assertEqual(base64.b64decode(files["project/main.py"]), b"print('hello')\n")

    def test_traversal_duplicate_and_symlink_rejected(self):
        link = zipfile.ZipInfo("link.py")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        for entries in [
            [("../main.py", "x")],
            [("/main.py", "x")],
            [("A.py", "x"), ("a.py", "y")],
            [("data", "x"), ("data/main.py", "y")],
            [("data/main.py", "x"), ("DATA", "y")],
            [(link, "/etc/passwd")],
            [("__dispatch_reference__.py", "x")],
        ]:
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                unpack(self.zip(entries))

    def test_comparison_checks_trial_identity_and_fixed_tolerance(self):
        original = {
            "ok": True,
            "items": [{"seed": 1, "value": {"x": 1.0}}, {"seed": 2, "value": {"x": 2.0}}],
        }
        candidates = [
            {"ok": True, "items": [original["items"][1]]},
            {"ok": True, "items": [original["items"][0]]},
        ]
        self.assertTrue(compare(original, candidates, [1, 2])["passed"])
        self.assertFalse(equivalent(True, 1))
        self.assertFalse(equivalent(float("nan"), float("nan")))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            compare(original, [original, original], [1, 2])

    def test_malformed_trial_items_are_rejected(self):
        with self.assertRaises(TypeError):
            compare({"ok": True, "items": [None]}, [], [1])

import json
import unittest

import httpx

from orchestrator.supervisor.models import LogSearch
from orchestrator.supervisor.sentry import SentryReader, SentryUnavailable


class SentryReaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_scopes_and_filters_with_pagination(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(
                200,
                json={"data": [{"id": "log-1", "job_id": "job-a", "message": "OOM"}]},
                headers={
                    "Link": '<https://sentry.io/next>; rel="next"; results="true"; cursor="abc:1:0"'
                },
            )

        reader = SentryReader(
            "test-token", "org", "project", transport=httpx.MockTransport(handler)
        )
        self.addAsyncCleanup(reader.close)
        result = await reader.search(
            "job-a", LogSearch(worker_id="worker-a", severity="error", text='x" OR job_id:other')
        )
        query = requests[0].url.params["query"]
        self.assertTrue(query.startswith('job_id:"job-a" AND worker_id:"worker-a"'))
        self.assertIn('severity:"error"', query)
        self.assertIn(json.dumps('*x" OR job_id:other*'), query)
        self.assertEqual(requests[0].url.params["dataset"], "logs")
        self.assertEqual(result["next_cursor"], "abc:1:0")
        self.assertTrue(result["has_more"])

    async def test_out_of_scope_rows_and_redirects_are_rejected(self):
        for response in [
            httpx.Response(200, json={"data": [{"job_id": "other"}]}),
            httpx.Response(302, headers={"Location": "https://other.example"}),
            httpx.Response(200, json={"unexpected": []}),
        ]:
            reader = SentryReader(
                "test-token",
                "org",
                "project",
                transport=httpx.MockTransport(lambda _, response=response: response),
            )
            try:
                with self.assertRaises(SentryUnavailable):
                    await reader.search("job-a", LogSearch())
            finally:
                await reader.close()

    async def test_error_lookup_keeps_job_and_project_scope(self):
        seen = []
        reader = SentryReader(
            "test-token",
            "org",
            "project",
            transport=httpx.MockTransport(
                lambda req: (seen.append(req), httpx.Response(200, json={"data": []}))[1]
            ),
        )
        self.addAsyncCleanup(reader.close)
        await reader.search("job-a", LogSearch(), dataset="errors", event_id="a" * 32)
        self.assertEqual(seen[0].url.params["dataset"], "errors")
        self.assertEqual(seen[0].url.params["project"], "project")
        self.assertIn('job_id:"job-a"', seen[0].url.params["query"])
        self.assertIn('id:"' + "a" * 32 + '"', seen[0].url.params["query"])

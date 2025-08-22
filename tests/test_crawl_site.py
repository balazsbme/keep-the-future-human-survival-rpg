import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ingest_opennebula_docs import crawl_site


class CrawlSiteTests(unittest.TestCase):
    @patch("ingest_opennebula_docs.requests.get")
    def test_crawl_collects_internal_links(self, mock_get):
        pages = {
            "https://docs.opennebula.io/7.0/": '<a href="page_b.html">b</a>',
            "https://docs.opennebula.io/7.0/page_b.html": '<a href="page_c.html">c</a>',
            "https://docs.opennebula.io/7.0/page_c.html": "",
        }

        def fake_get(url, timeout=10):
            class Resp:
                def __init__(self, text):
                    self.text = text
                    self.status_code = 200
                def raise_for_status(self):
                    pass
            return Resp(pages[url])

        mock_get.side_effect = fake_get
        urls = crawl_site("https://docs.opennebula.io/7.0/")
        self.assertEqual(
            set(urls),
            {
                "https://docs.opennebula.io/7.0/",
                "https://docs.opennebula.io/7.0/page_b.html",
                "https://docs.opennebula.io/7.0/page_c.html",
            },
        )

    @patch("ingest_opennebula_docs.logging")
    @patch("ingest_opennebula_docs.requests.get")
    def test_crawl_logs_start_and_end(self, mock_get, mock_logging):
        pages = {
            "https://docs.opennebula.io/7.0/": '<a href="page_b.html">b</a>',
            "https://docs.opennebula.io/7.0/page_b.html": '<a href="page_c.html">c</a>',
            "https://docs.opennebula.io/7.0/page_c.html": "",
        }

        def fake_get(url, timeout=10):
            class Resp:
                def __init__(self, text):
                    self.text = text
                    self.status_code = 200
                def raise_for_status(self):
                    pass
            return Resp(pages[url])

        mock_get.side_effect = fake_get
        crawl_site("https://docs.opennebula.io/7.0/")

        mock_logging.info.assert_any_call(
            "Starting crawl for %s", "https://docs.opennebula.io/7.0/"
        )
        mock_logging.info.assert_any_call(
            "Completed crawl for %s with %d urls",
            "https://docs.opennebula.io/7.0/",
            3,
        )

if __name__ == "__main__":
    unittest.main()

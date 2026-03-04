import unittest

from api_server import build_uvicorn_log_config


class ApiServerLogConfigTests(unittest.TestCase):
    def test_default_log_formatter_contains_timestamp(self):
        config = build_uvicorn_log_config()
        default_formatter = config["formatters"]["default"]
        self.assertIn("%(asctime)s", default_formatter["fmt"])
        self.assertEqual(default_formatter["datefmt"], "%Y-%m-%d %H:%M:%S")

    def test_access_log_formatter_contains_timestamp(self):
        config = build_uvicorn_log_config()
        access_formatter = config["formatters"]["access"]
        self.assertIn("%(asctime)s", access_formatter["fmt"])
        self.assertEqual(access_formatter["datefmt"], "%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    unittest.main()

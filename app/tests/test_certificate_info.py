import unittest
import certificate_info

class CertificateInfoTestCase(unittest.TestCase):
    def test_get_certificate_issued_to(self):
        hostname = "www.google.com"
        issued_to = certificate_info.get_certificate_issued_to(hostname)
        self.assertIsNotNone(issued_to)
        self.assertIsInstance(issued_to, str)

    def test_get_certificate_issued_by(self):
        hostname = "www.google.com"
        issued_by = certificate_info.get_certificate_issued_by(hostname)
        self.assertIsNotNone(issued_by)
        self.assertIsInstance(issued_by, str)

if __name__ == '__main__':
    unittest.main()

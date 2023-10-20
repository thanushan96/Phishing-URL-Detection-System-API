import unittest
from domain_info import domaincreatedate, domainexpiredate, ageofdomain1

class DomainInfoTestCase(unittest.TestCase):
    def test_domaincreatedate_valid(self):
        url = "https://www.google.com/"
        creation_date = domaincreatedate(url)
        self.assertNotEqual(creation_date, "No domain information for this URL")

    def test_domaincreatedate_invalid(self):
        url = "https://invalid.invalid/"
        creation_date = domaincreatedate(url)
        self.assertEqual(creation_date, "No domain information for this URL")

    def test_domainexpiredate_valid(self):
        url = "https://www.google.com/"
        expiration_date = domainexpiredate(url)
        self.assertNotEqual(expiration_date, "No domain information for this URL")

    def test_domainexpiredate_invalid(self):
        url = "https://invalid.invalid/"
        expiration_date = domainexpiredate(url)
        self.assertEqual(expiration_date, "No domain information for this URL")

    def test_ageofdomain1_valid(self):
        url = "https://www.google.com/"
        age_of_domain = ageofdomain1(url)
        self.assertNotEqual(age_of_domain, -1)


    def test_ageofdomain1_invalid(self):
        url = "https://invalid.invalid/"
        age_of_domain = ageofdomain1(url)
        self.assertEqual(age_of_domain, -1)

if __name__ == '__main__':
    unittest.main()

import unittest
from app import app  
import json

class PredictTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        app.config['TESTING'] = True

    def test_predict_valid_url(self):
        sample_url = 'http://owaloginno.16mb.com/'  
        data = {'url': sample_url}
        response = self.app.post('/predict', data=data)

        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)

        self.assertEqual(response_data['status_value'], 'URL is in a valid format:')
        self.assertEqual(response_data['urlstatus'], 'Phishing URL')

    def test_predict_invalid_url(self):
        sample_url = 'google.com'  
        data = {'url': sample_url}
        response = self.app.post('/predict', data=data)

        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)

       
        self.assertEqual(response_data['status_value'], 'Enter URL in a valid format (e.g., \'https://www.example.com\')')

if __name__ == '__main__':
    unittest.main()




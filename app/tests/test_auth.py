import unittest
from app import app
import json
import random
import string  

class AuthTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        app.config['TESTING'] = True

    def random_username(self):
        return ''.join(random.choice(string.ascii_letters) for _ in range(8))  

    def test_register(self):
        username = self.random_username()  
        data = {
            'username': username,
            'email': f'{username}@example.com',
            'password': 'testpassword'
        }

        response = self.app.post('/register', data=json.dumps(data), content_type='application/json')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Registration successful', response.data)

    def test_login_successful(self):
        # Register a test user
        username = self.random_username()  
        data = {
            'username': username,
            'email': f'{username}@example.com',
            'password': 'testpassword'
        }

        self.app.post('/register', data=json.dumps(data), content_type='application/json')

        # Attempt login with the registered user
        data = {
            'username': username,
            'password': 'testpassword'
        }

        response = self.app.post('/login', data=json.dumps(data), content_type='application/json')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login successful', response.data)

    def test_login_failed(self):
        # Register a test user
        username = self.random_username()  
        data = {
            'username': username,
            'email': f'{username}@example.com',
            'password': 'testpassword'
        }

        self.app.post('/register', data=json.dumps(data), content_type='application/json')

        # Attempt login with the wrong password
        data = {
            'username': username,
            'password': 'wrongpassword'
        }

        response = self.app.post('/login', data=json.dumps(data), content_type='application/json')

        self.assertEqual(response.status_code, 401)
        self.assertIn(b'Login failed', response.data)

if __name__ == '__main__':
    unittest.main()

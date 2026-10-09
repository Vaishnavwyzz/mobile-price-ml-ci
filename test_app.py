import unittest
from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_low_specification_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "battery_power": 500,
                "blue": 0,
                "clock_speed": 0.5,
                "dual_sim": 0,
                "fc": 1,
                "four_g": 0,
                "int_memory": 8,
                "m_dep": 0.5,
                "mobile_wt": 190,
                "n_cores": 2,
                "pc": 2,
                "px_height": 300,
                "px_width": 500,
                "ram": 512,
                "sc_h": 8,
                "sc_w": 4,
                "talk_time": 6,
                "three_g": 0,
                "touch_screen": 0,
                "wifi": 0
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(response.get_json()["prediction"], [0, 1, 2, 3])

    def test_high_specification_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "battery_power": 1900,
                "blue": 1,
                "clock_speed": 2.5,
                "dual_sim": 1,
                "fc": 8,
                "four_g": 1,
                "int_memory": 64,
                "m_dep": 0.5,
                "mobile_wt": 120,
                "n_cores": 8,
                "pc": 16,
                "px_height": 1200,
                "px_width": 2000,
                "ram": 3500,
                "sc_h": 17,
                "sc_w": 9,
                "talk_time": 18,
                "three_g": 1,
                "touch_screen": 1,
                "wifi": 1
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(response.get_json()["prediction"], [0, 1, 2, 3])

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "battery_power": 1000,
                "blue": 1
            }
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())


if __name__ == "__main__":
    unittest.main()

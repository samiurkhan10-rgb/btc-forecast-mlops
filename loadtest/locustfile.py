"""Locust load test for the BTC forecast API (used to trigger the HPA).

Headless example (PowerShell, from the repo root, with `minikube tunnel` running):
    .venv\\Scripts\\locust -f loadtest\\locustfile.py --host http://btc-forecast.local `
        --headless -u 60 -r 5 -t 4m `
        --csv loadtest\\results\\run --html loadtest\\results\\report.html

/predict runs the recursive XGBoost forecast (CPU-bound), so it is weighted 3x more than /health.
"""

import random

from locust import HttpUser, between, task


class ForecastUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(3)
    def predict(self):
        days = random.choice([7, 7, 7, 14, 30])  # mostly the default 7-day horizon
        with self.client.post(
            "/predict", json={"forecast_days": days}, name="/predict", catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            elif len(resp.json().get("predictions", [])) != days:
                resp.failure("wrong number of predictions")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")

"""
Unit tests for Flask Web Dashboard REST API and Endpoints.
"""

import unittest
import json
import sys
import os

from web_dashboard.app import app


class TestWebDashboardAPI(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_page(self):
        res = self.app.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'SkyGuard PhysioSense', res.data)

    def test_api_stations(self):
        res = self.app.get('/api/stations')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data['status'], 'success')
        self.assertGreaterEqual(len(data['stations']), 4)
        self.assertIn('lat', data['stations'][0])

    def test_api_live_telemetry(self):
        res = self.app.get('/api/telemetry/live?station_id=AWS-01-ISRIKA1-Bheemili')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('temperature', data['current'])
        self.assertIn('ci95_lower', data['current']['temperature'])
        self.assertIn('chart_series', data)

    def test_api_history(self):
        res = self.app.get('/api/telemetry/history?station_id=AWS-01-ISRIKA1-Bheemili')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('timestamps', data)
        self.assertIn('temperature', data)

    def test_api_alerts(self):
        res = self.app.get('/api/alerts?station_id=AWS-01-ISRIKA1-Bheemili&severity=ALL')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data['status'], 'success')
        self.assertIsInstance(data['alerts'], list)

    def test_api_health(self):
        res = self.app.get('/api/health?station_id=AWS-01-ISRIKA1-Bheemili')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('temperature', data['health_metrics'])
        self.assertIn('health_score', data['health_metrics']['temperature'])

    def test_api_metrics(self):
        res = self.app.get('/api/metrics')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('throughput_rps', data)

    def test_api_inject_anomaly(self):
        payload = {'station_id': 'AWS-01-ISRIKA1-Bheemili', 'type': 'temp_spike'}
        res = self.app.post('/api/inject_anomaly', json=payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data['status'], 'success')

    def test_api_export_csv_and_json(self):
        res_csv = self.app.get('/api/export/csv?station_id=AWS-01-ISRIKA1-Bheemili')
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn('text/csv', res_csv.headers['Content-Type'])

        res_json = self.app.get('/api/export/json?station_id=AWS-01-ISRIKA1-Bheemili')
        self.assertEqual(res_json.status_code, 200)
        self.assertIn('application/json', res_json.headers['Content-Type'])


if __name__ == '__main__':
    unittest.main()

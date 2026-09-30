"""Verify notification retry behavior without sending real messages or using GPUs."""
import datetime as dt
import unittest
from unittest.mock import patch

import report_tick as tick


class ReportTimerTests(unittest.TestCase):
    def test_publish_failure_does_not_repeat_successful_chat_notification(self):
        calls = []
        first = True

        def run(argv, timeout):
            nonlocal first
            calls.append(argv)
            if argv[0] == 'bash' and first:
                first = False
                return {'returncode': 1}
            return {'returncode': 0}

        config = {'codex_bin': '/example/codex', 'thread_id': 'test-only'}
        receipt = {}
        self.assertFalse(tick.deliver(config, receipt, 50, run))
        self.assertTrue(tick.deliver(config, receipt, 50, run))
        self.assertEqual(sum(c[0] == '/example/codex' for c in calls), 1)
        self.assertEqual(sum(c[0] == 'bash' for c in calls), 2)
        self.assertTrue(tick.deliver(config, receipt, 51, run))
        self.assertEqual(sum(c[0] == '/example/codex' for c in calls), 2)

    def test_failed_notification_retries_without_republishing(self):
        calls = []

        def run(argv, timeout):
            calls.append(argv)
            return {'returncode': 1 if len(calls) == 2 else 0}

        receipt = {}
        config = {'codex_bin': '/example/codex', 'thread_id': 'test-only'}
        self.assertFalse(tick.deliver(config, receipt, 50, run))
        self.assertTrue(tick.deliver(config, receipt, 50, run))
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[2][1], 'queue')

    def test_calendar_is_six_hours_and_crosses_midnight(self):
        first = dt.datetime(2026, 9, 28, 8, 31, tzinfo=tick.TIMEZONE)
        due = dt.datetime.fromisoformat(tick.next_due(first.timestamp()))
        self.assertEqual((due.hour, due.minute), (14, 30))
        later = dt.datetime.fromisoformat(tick.next_due(due.timestamp()))
        self.assertEqual((later-due).total_seconds(), 21600)
        final = dt.datetime.fromisoformat(tick.next_due(later.timestamp()))
        self.assertEqual((final.day, final.hour, final.minute), (29, 2, 30))

    def test_timer_stops_only_after_both_verified_50k_results(self):
        with patch.object(tick, 'read', return_value={'status': 'complete'}):
            with patch.object(tick, 'comparison', return_value={'status': 'pending'}):
                self.assertFalse(tick.finished())
            with patch.object(tick, 'comparison', side_effect=[{'status': 'matched'}, {'status': 'pending'}]):
                self.assertFalse(tick.finished())
            with patch.object(tick, 'comparison', return_value={'status': 'matched'}):
                self.assertTrue(tick.finished())

    def test_timer_keeps_reporting_while_600_epoch_decision_is_pending(self):
        def read(path, default=None):
            if path.name == 'experiment_plan.json':
                return {'extension_600': {'status': 'pending_200_results_and_decision'}}
            return {'status': 'complete'}
        with patch.object(tick, 'read', side_effect=read), \
             patch.object(tick, 'comparison', return_value={'status':'matched'}):
            self.assertFalse(tick.finished())


if __name__ == '__main__':
    unittest.main()

"""Offline tests for --fc-mode iffc-state-ooda (no server, no model)."""

from benchmark import smoke_state_ooda


def test_state_ooda_offline_smoke():
    smoke_state_ooda.run_offline_smoke()


def test_state_ooda_mock_loop_state_before_card_and_stop_gate():
    smoke_state_ooda.run_mock_loop()

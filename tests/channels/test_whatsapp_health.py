import asyncio

import pytest

from nanobot.channels.whatsapp_health import WhatsAppHealthMonitor


def _collector():
    sent: list[str] = []

    async def _send(text: str) -> None:
        sent.append(text)

    return _send, sent


class _Clock:
    """Manual clock so debounce/window logic is deterministic."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.mark.asyncio
async def test_logged_out_alerts_immediately():
    send, sent = _collector()
    mon = WhatsAppHealthMonitor(send_alert=send)
    await mon.on_status_change("connected")
    await mon.on_status_change("logged_out")
    assert len(sent) == 1
    assert "logged out" in sent[0].lower()


@pytest.mark.asyncio
async def test_recovery_alert_after_outage():
    send, sent = _collector()
    mon = WhatsAppHealthMonitor(send_alert=send)
    await mon.on_status_change("connected")
    await mon.on_status_change("logged_out")   # alert #1
    await mon.on_status_change("connected")    # all-clear
    assert len(sent) == 2
    assert "reconnected" in sent[1].lower()


@pytest.mark.asyncio
async def test_no_recovery_alert_without_prior_outage():
    """A connect with no preceding alerted-outage shouldn't ping."""
    send, sent = _collector()
    mon = WhatsAppHealthMonitor(send_alert=send)
    await mon.on_status_change("connected")
    assert sent == []


@pytest.mark.asyncio
async def test_disabled_when_no_send_alert():
    """Opt-in: send_alert=None → never pings, even on logged_out."""
    mon = WhatsAppHealthMonitor(send_alert=None)
    await mon.on_status_change("connected")
    await mon.on_status_change("logged_out")
    # No exception, no crash — and nothing to assert delivered (no sender).


@pytest.mark.asyncio
async def test_debounce_suppresses_repeat_logged_out():
    send, sent = _collector()
    clock = _Clock()
    mon = WhatsAppHealthMonitor(send_alert=send, alert_debounce_s=300, clock=clock)
    await mon.on_status_change("connected")
    await mon.on_status_change("logged_out")   # alert
    await mon.on_status_change("connected")    # recovery (force, always sends)
    clock.advance(10)
    await mon.on_status_change("logged_out")   # within debounce window → suppressed
    logged_out_alerts = [s for s in sent if "logged out" in s.lower()]
    assert len(logged_out_alerts) == 1


@pytest.mark.asyncio
async def test_disconnect_alerts_after_grace_if_still_down():
    send, sent = _collector()
    mon = WhatsAppHealthMonitor(send_alert=send, disconnect_alert_after_s=0.02)
    await mon.on_status_change("connected")
    await mon.on_status_change("disconnected")
    await asyncio.sleep(0.05)  # let the grace timer fire
    assert any("disconnected" in s.lower() for s in sent)


@pytest.mark.asyncio
async def test_transient_disconnect_does_not_alert():
    """Disconnect then reconnect within the grace period → no alert."""
    send, sent = _collector()
    mon = WhatsAppHealthMonitor(send_alert=send, disconnect_alert_after_s=0.1)
    await mon.on_status_change("connected")
    await mon.on_status_change("disconnected")
    await mon.on_status_change("connected")   # cancels the grace timer
    await asyncio.sleep(0.15)
    assert sent == []  # neither a down-alert nor a recovery (never alerted down)


@pytest.mark.asyncio
async def test_flap_detection_alerts():
    """Three *disconnects* within the window → unstable alert. Reconnects
    don't count toward the flap total."""
    send, sent = _collector()
    clock = _Clock()
    mon = WhatsAppHealthMonitor(
        send_alert=send, flap_threshold=3, flap_window_s=60,
        disconnect_alert_after_s=999, clock=clock,
    )
    for _ in range(3):
        await mon.on_status_change("connected")
        clock.advance(1)
        await mon.on_status_change("disconnected")
        clock.advance(1)
    assert any("unstable" in s.lower() for s in sent)


@pytest.mark.asyncio
async def test_clean_outage_recovery_not_flagged_as_flap():
    """A single down+up cycle must NOT trip the flap detector."""
    send, sent = _collector()
    mon = WhatsAppHealthMonitor(send_alert=send, flap_threshold=3)
    await mon.on_status_change("connected")
    await mon.on_status_change("logged_out")
    await mon.on_status_change("connected")
    assert not any("unstable" in s.lower() for s in sent)


@pytest.mark.asyncio
async def test_duplicate_status_events_ignored():
    """neonize emits doubled events; identical consecutive status is a no-op."""
    send, sent = _collector()
    mon = WhatsAppHealthMonitor(send_alert=send)
    await mon.on_status_change("logged_out")
    await mon.on_status_change("logged_out")  # duplicate
    assert len([s for s in sent if "logged out" in s.lower()]) == 1


@pytest.mark.asyncio
async def test_telemetry_hook_fires_on_transition():
    send, _ = _collector()
    events: list[tuple] = []
    mon = WhatsAppHealthMonitor(
        send_alert=send,
        telemetry=lambda prev, new, dur: events.append((prev, new)),
    )
    await mon.on_status_change("connected")   # prev=None → no telemetry
    await mon.on_status_change("disconnected")
    assert ("connected", "disconnected") in events

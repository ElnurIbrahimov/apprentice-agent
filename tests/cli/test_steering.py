"""Tests for mid-turn steering queue."""
import pytest
from aura.cli.steering import SteeringQueue, create_steering_indicator, create_follow_up_indicator

def test_push_and_pop():
    q = SteeringQueue()
    q.push("do X instead")
    q.push("also check Y")
    msgs = q.pop_all()
    assert len(msgs) == 2
    assert msgs[0] == "do X instead"
    assert msgs[1] == "also check Y"

def test_pop_empty():
    q = SteeringQueue()
    assert q.pop_all() == []

def test_has_messages():
    q = SteeringQueue()
    assert not q.has_messages()
    q.push("hello")
    assert q.has_messages()
    q.pop_all()
    assert not q.has_messages()

def test_follow_up():
    q = SteeringQueue()
    q.push_follow_up("now do this next")
    assert q.has_follow_up()
    msg = q.pop_follow_up()
    assert msg == "now do this next"
    assert not q.has_follow_up()

def test_max_queued():
    q = SteeringQueue(max_queued=3)
    for i in range(5):
        q.push(f"msg{i}")
    msgs = q.pop_all()
    assert len(msgs) == 3
    assert msgs[0] == "msg2"  # oldest dropped

def test_clear():
    q = SteeringQueue()
    q.push("a")
    q.push_follow_up("b")
    q.clear()
    assert not q.has_messages()
    assert not q.has_follow_up()

def test_format_injection_single():
    q = SteeringQueue()
    q.push("focus on security")
    result = q.format_injection()
    assert "focus on security" in result
    assert "SYSTEM NOTE" in result

def test_format_injection_multiple():
    q = SteeringQueue()
    q.push("do A")
    q.push("also B")
    result = q.format_injection()
    assert "do A" in result
    assert "also B" in result
    assert "SYSTEM NOTE" in result

def test_format_injection_empty():
    q = SteeringQueue()
    assert q.format_injection() is None

def test_steering_indicator_empty():
    q = SteeringQueue()
    assert create_steering_indicator(q) == ""

def test_steering_indicator_one():
    q = SteeringQueue()
    q.push("msg")
    indicator = create_steering_indicator(q)
    assert "1" in indicator

def test_follow_up_indicator():
    q = SteeringQueue()
    assert create_follow_up_indicator(q) == ""
    q.push_follow_up("next")
    assert "follow-up" in create_follow_up_indicator(q)

def test_thread_safety():
    """Basic thread safety test — push from multiple threads."""
    import threading
    q = SteeringQueue(max_queued=100)

    def pusher(n):
        for i in range(10):
            q.push(f"thread{n}_msg{i}")

    threads = [threading.Thread(target=pusher, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    msgs = q.pop_all()
    assert len(msgs) <= 100
    assert len(msgs) >= 1  # At least some messages made it

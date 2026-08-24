import live


def test_new_text_with_no_previous_text():
    assert live._new_text("", "hello there") == "hello there"


def test_new_text_returns_only_the_new_part():
    prev = "so michael tell me more about"
    new = "so michael tell me more about you today"
    assert live._new_text(prev, new) == "you today"


def test_new_text_returns_full_text_when_no_overlap():
    assert live._new_text("xyz123", "qrs789") == "qrs789"


def test_stream_reconciler_emits_only_new_pieces_across_windows():
    reconciler = live._StreamReconciler("user")
    first = reconciler.step("hey michael", elapsed=1.0)
    second = reconciler.step("hey michael how are you", elapsed=2.0)
    assert first == "hey michael"
    assert second == "how are you"
    assert len(reconciler.segments) == 2
    assert reconciler.segments[0]["speaker"] == "user"


def test_stream_reconciler_emits_nothing_for_repeated_window():
    reconciler = live._StreamReconciler("system")
    reconciler.step("same text", elapsed=1.0)
    second = reconciler.step("same text", elapsed=2.0)
    assert second == ""
    assert len(reconciler.segments) == 1

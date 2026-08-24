import json
from contextlib import contextmanager
from unittest.mock import patch

import transcribe as t


def test_format_timestamp_under_an_hour():
    assert t.format_timestamp(65) == "01:05"


def test_format_timestamp_over_an_hour():
    assert t.format_timestamp(3665) == "01:01:05"


def test_format_srt_timestamp():
    assert t.format_srt_timestamp(65.25) == "00:01:05,250"


def test_group_by_pause_merges_close_segments():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.5, "end": 2.5, "text": "world"},
    ]
    groups = t.group_by_pause(segments, gap_seconds=4.5, max_block_seconds=45)
    assert len(groups) == 1
    assert groups[0]["text"] == "hello world"


def test_group_by_pause_breaks_on_silence_gap():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 10.0, "end": 11.0, "text": "world"},
    ]
    groups = t.group_by_pause(segments, gap_seconds=4.5, max_block_seconds=45)
    assert len(groups) == 2


def test_group_by_pause_breaks_on_speaker_change():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hey", "speaker": "user"},
        {"start": 1.2, "end": 2.0, "text": "hi", "speaker": "system"},
    ]
    groups = t.group_by_pause(segments, gap_seconds=4.5, max_block_seconds=45)
    assert len(groups) == 2
    assert groups[0]["speaker"] == "user"
    assert groups[1]["speaker"] == "system"


def test_group_by_pause_breaks_on_max_block_length():
    segments = [
        {"start": 0.0, "end": 30.0, "text": "one"},
        {"start": 30.5, "end": 50.0, "text": "two"},
    ]
    groups = t.group_by_pause(segments, gap_seconds=4.5, max_block_seconds=45)
    assert len(groups) == 2


def test_group_by_pause_skips_empty_text():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "  "},
        {"start": 1.0, "end": 2.0, "text": "real"},
    ]
    groups = t.group_by_pause(segments, gap_seconds=4.5, max_block_seconds=45)
    assert len(groups) == 1
    assert groups[0]["text"] == "real"


def test_render_markdown_includes_speaker_tag():
    segments = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "user"}]
    md = t.render_markdown("title", segments)
    assert "**[00:00 - 00:01] (user)** hi" in md


def test_render_markdown_omits_speaker_tag_when_absent():
    segments = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
    md = t.render_markdown("title", segments)
    assert "(" not in md.split("**")[-1].split("hi")[0]


def test_render_srt_numbers_cues(monkeypatch):
    monkeypatch.setattr(t, "PAUSE_GAP_SECONDS", 1.0)
    monkeypatch.setattr(t, "MAX_BLOCK_SECONDS", 45)
    segments = [
        {"start": 0.0, "end": 1.0, "text": "one"},
        {"start": 5.0, "end": 6.0, "text": "two"},
    ]
    srt = t.render_srt(segments)
    assert srt.startswith("1\n")
    assert "\n2\n" in srt
    assert "-->" in srt


def test_suggest_filename_picks_frequent_words_in_first_seen_order():
    segments = [
        {"text": "wuthering waves boss fight build guide"},
        {"text": "wuthering waves boss fight is hard"},
    ]
    name = t.suggest_filename = t._suggest_filename(segments, max_words=3)
    assert name == "wuthering_waves_boss"


def test_suggest_filename_returns_none_for_empty_transcript():
    assert t._suggest_filename([{"text": ""}]) is None


def test_suggest_filename_returns_none_when_only_stopwords():
    assert t._suggest_filename([{"text": "the a an is"}]) is None


def test_strip_invalid_filename_chars():
    assert t._strip_invalid_filename_chars('a<b>c:d"e/f\\g|h?i*j') == "abcdefghij"


def test_word_count():
    assert t._word_count("Hey, I'm Michael. I'm 34 years old.") == 7


def test_slugify_title():
    assert t._slugify_title("Weekly Standup Meeting") == "weekly_standup_meeting"


def test_resolve_file_path_finds_existing_file(tmp_path, monkeypatch):
    real_file = tmp_path / "clip.mp4"
    real_file.write_text("data")
    monkeypatch.setattr(t, "VIDEO_EXTENSIONS", {".mp4"})
    result = t._resolve_file_path(tmp_path / "clip")
    assert result == real_file


def test_resolve_file_path_returns_original_when_nothing_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(t, "VIDEO_EXTENSIONS", {".mp4"})
    missing = tmp_path / "missing"
    assert t._resolve_file_path(missing) == missing


@contextmanager
def _fake_response(body: dict):
    class _Resp:
        status = 200

        def read(self):
            return json.dumps(body).encode("utf-8")

    yield _Resp()


def test_llm_punctuate_batch_returns_corrected_lines():
    reply = {"message": {"content": "1: Hello there.\n2: How are you?"}}
    with patch("urllib.request.urlopen", return_value=_fake_response(reply)):
        result = t._llm_punctuate_batch(["hello there", "how are you"])
    assert result == ["Hello there.", "How are you?"]


def test_llm_punctuate_batch_returns_none_on_mismatched_line_count():
    reply = {"message": {"content": "1: Hello there."}}
    with patch("urllib.request.urlopen", return_value=_fake_response(reply)):
        result = t._llm_punctuate_batch(["hello there", "how are you"])
    assert result is None


def test_llm_punctuate_batch_returns_none_on_network_error():
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        assert t._llm_punctuate_batch(["hello"]) is None


def test_restore_punctuation_keeps_original_on_word_count_mismatch(monkeypatch):
    monkeypatch.setattr(t, "_llm_runtime_available", True)
    monkeypatch.setattr(t, "LLM_ENABLED", True)
    segments = [{"text": "living in seattle"}]
    with patch.object(t, "_llm_punctuate_batch", return_value=["and living in Seattle"]):
        t._restore_punctuation("test", segments)
    assert segments[0]["text"] == "living in seattle"


def test_restore_punctuation_applies_matching_word_count(monkeypatch):
    monkeypatch.setattr(t, "_llm_runtime_available", True)
    monkeypatch.setattr(t, "LLM_ENABLED", True)
    segments = [{"text": "hello there"}]
    with patch.object(t, "_llm_punctuate_batch", return_value=["Hello there."]):
        t._restore_punctuation("test", segments)
    assert segments[0]["text"] == "Hello there."


def test_restore_punctuation_noop_when_llm_unusable(monkeypatch):
    monkeypatch.setattr(t, "_llm_runtime_available", False)
    monkeypatch.setattr(t, "LLM_ENABLED", True)
    segments = [{"text": "hello there"}]
    t._restore_punctuation("test", segments)
    assert segments[0]["text"] == "hello there"


def test_generate_title_skips_short_transcript(monkeypatch):
    monkeypatch.setattr(t, "_llm_runtime_available", True)
    monkeypatch.setattr(t, "LLM_AUTO_TITLE", True)
    segments = [{"text": "hey there"}]
    assert t._generate_title(segments) is None


def test_generate_title_returns_none_when_unusable(monkeypatch):
    monkeypatch.setattr(t, "_llm_runtime_available", False)
    monkeypatch.setattr(t, "LLM_AUTO_TITLE", True)
    segments = [{"text": "this transcript has plenty of words in it to pass the length check"}]
    assert t._generate_title(segments) is None


def test_generate_title_slugifies_llm_response(monkeypatch):
    monkeypatch.setattr(t, "_llm_runtime_available", True)
    monkeypatch.setattr(t, "LLM_AUTO_TITLE", True)
    segments = [{"text": "this transcript has plenty of words in it to pass the length check"}]
    reply = {"message": {"content": "Weekly Standup Meeting"}}
    with patch("urllib.request.urlopen", return_value=_fake_response(reply)):
        assert t._generate_title(segments) == "weekly_standup_meeting"

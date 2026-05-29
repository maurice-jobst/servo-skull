"""Test _utils module."""
import pytest
from pathlib import Path
from servo_skull._utils import calculate_checksum, retry, safe_json_read, safe_json_write


def test_calculate_checksum():
    """Test checksum calculation."""
    data = b"test data"
    checksum = calculate_checksum(data)
    assert len(checksum) == 64  # SHA256 hex is 64 chars
    assert checksum == calculate_checksum(data)  # Deterministic


def test_retry_succeeds_on_first_attempt():
    """Test retry decorator succeeds immediately."""
    call_count = 0

    @retry(max_attempts=3)
    def test_func():
        nonlocal call_count
        call_count += 1
        return "success"

    result = test_func()
    assert result == "success"
    assert call_count == 1


def test_retry_succeeds_on_second_attempt():
    """Test retry decorator retries on failure."""
    call_count = 0

    @retry(max_attempts=3, delay=0.01)
    def test_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("First attempt fails")
        return "success"

    result = test_func()
    assert result == "success"
    assert call_count == 2


def test_retry_exhausts_attempts():
    """Test retry decorator exhausts max attempts and re-raises exception."""
    @retry(max_attempts=2, delay=0.01)
    def test_func():
        raise ValueError("Always fails")

    with pytest.raises(ValueError, match="Always fails"):
        test_func()


def test_safe_json_write_read(tmp_path):
    """Test JSON write/read round-trip."""
    test_file = tmp_path / "test.json"
    data = {"key": "value", "number": 42}

    assert safe_json_write(test_file, data) is True
    assert test_file.exists()

    read_data = safe_json_read(test_file)
    assert read_data == data


def test_safe_json_read_nonexistent():
    """Test safe_json_read returns None for nonexistent file."""
    result = safe_json_read(Path("/nonexistent/file.json"))
    assert result is None


def test_count_syllables():
    """Test syllable counter estimates."""
    from servo_skull._utils import count_syllables
    assert count_syllables("cat") == 1
    assert count_syllables("apple") == 2
    assert count_syllables("banana") == 3
    assert count_syllables("beautiful") == 3
    assert count_syllables("a") == 1
    assert count_syllables("") == 0


def test_calculate_flesch_reading_ease():
    """Test Flesch Reading Ease score calculation."""
    from servo_skull._utils import calculate_flesch_reading_ease
    # Very simple text: "The cat sat on the mat."
    # 6 words, 6 syllables, 1 sentence.
    # FRE = 206.835 - 1.015 * (6/1) - 84.6 * (6/6) = 206.835 - 6.09 - 84.6 = 116.14
    score_simple = calculate_flesch_reading_ease("The cat sat on the mat.")
    assert score_simple > 100.0

    # Test exclusion of code blocks and technical acronyms
    text_with_code = """
    Here is a description of the system.
    ```python
    def complex_code_block():
        some_complex_algorithm_with_many_words_and_syllables()
    ```
    We must use the SKU100 module with API and SDK integration.
    """
    # The code block and SKU100, API, SDK should be ignored in the count.
    score_complex = calculate_flesch_reading_ease(text_with_code)
    # The score should be in a reasonable range
    assert 0.0 < score_complex < 100.0


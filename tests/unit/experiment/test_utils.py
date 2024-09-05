import pytest

from nomadic.experiment.utils import convert_string_to_list


def test_convert_string_to_list():
    # Test with a list of strings representing floats
    assert convert_string_to_list("['1.1', '2.2', '3.3']") == [1.1, 2.2, 3.3]

    # Test with a list of floats
    assert convert_string_to_list("[1.1, 2.2, 3.3]") == [1.1, 2.2, 3.3]

    # Test with a list of ints
    assert convert_string_to_list("[1, 2, 3]") == [1, 2, 3]

    # Test with a mixed list of strings that can be converted to floats and ints
    assert convert_string_to_list("['1', '2.2', '3']") == [1.0, 2.2, 3.0]

    # Test with a list of strings
    assert convert_string_to_list("['a', 'b', 'c']") == ["a", "b", "c"]

    # Test with a non-list string
    with pytest.raises(ValueError, match="The provided string is not a list."):
        convert_string_to_list("'not a list'")

    # Test with an empty list
    assert convert_string_to_list("[]") == []

    # Test with a list containing a dictionary
    assert convert_string_to_list("[{'a': 1}, {'b': 2}]") == [{"a": 1}, {"b": 2}]

    # Test with a list containing nested lists
    assert convert_string_to_list("[[1, 2], [3, 4]]") == [[1, 2], [3, 4]]

    # Test with a list of boolean values
    assert convert_string_to_list("[True, False]") == [True, False]


if __name__ == "__main__":
    pytest.main()

import ast
from typing import Any, List


def convert_string_to_list(s: str) -> List[Any]:
    try:
        # Use literal_eval to safely evaluate the string to a Python object
        result = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        raise ValueError("Invalid string representation of a list.")
    if isinstance(result, list):
        try:
            # Check if all elements are strings that can be converted to floats
            if all(
                isinstance(x, str) and x.replace(".", "", 1).isdigit()
                for x in result
            ):
                return [float(x) for x in result]
            # Check if all elements are already floats or ints
            elif all(isinstance(x, float) for x in result):
                return [float(x) for x in result]
            elif all(isinstance(x, int) for x in result):
                return [int(x) for x in result]
            # Otherwise, return the list as is (assuming it's a list of strings)
            else:
                return result
        except (ValueError, SyntaxError):
            raise ValueError("Invalid string representation of a list.")
    else:
        raise ValueError("The provided string is not a list.")

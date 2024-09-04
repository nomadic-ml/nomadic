import json
from datetime import datetime


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def datetime_decoder(dct):
    for k, v in dct.items():
        if isinstance(v, str):
            try:
                dct[k] = datetime.fromisoformat(v)
            except ValueError:
                pass
    return dct


# Usage for serialization (when saving to database):
def serialize_data(data):
    return json.dumps(data, cls=DateTimeEncoder)


# Usage for deserialization (when loading from database):
def deserialize_data(json_str):
    return json.loads(json_str, object_hook=datetime_decoder)


def convert_null_to_none(data):
    if isinstance(data, dict):
        return {k: convert_null_to_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_null_to_none(item) for item in data]
    elif data == "null":
        return None
    else:
        return data

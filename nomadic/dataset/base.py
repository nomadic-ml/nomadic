import json
from datetime import datetime
import io
from typing import Any, Dict, List, Optional

import boto3
from pydantic import BaseModel, Field


class Dataset(BaseModel):
    """Run result"""

    content: Optional[List[Dict]]
    continuous_source: Optional[Dict]
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Metadata"
    )

    def model_post_init(self, __context):
        if not (self.content and self.continuous_source):
            raise Exception("Either dataset's `content` or `continuous_source` field have to be defined")

    def get_continuous_content(self, cutoff_datetime) -> List[Dict]:
        bucket_name, json_file_key = self.continuous_source["bucket_name"], self.continuous_source["json_file_key"]
        # Download the JSON file from S3
        data = _download_json_from_s3(bucket_name, json_file_key)

        if data:
            # Filter the entries by the cutoff date
            filtered_entries = _filter_entries_by_date(data, cutoff_datetime)

            # Process the filtered entries (e.g., save to file, further processing, etc.)
            if filtered_entries:
                print(f"Found {len(filtered_entries)} entries after the cutoff date:")
                # Save the filtered data to a file or process it further
                with open('filtered_data.json', 'w') as outfile:
                    return [json.dump(filtered_entries, outfile, indent=4)]
                print("Filtered data saved to 'filtered_data.json'")
            else:
                print("No entries found after the cutoff date.")
        else:
            print("No data was downloaded from S3.")


# Function to download JSON file from S3
def _download_json_from_s3(bucket, key):
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        # Get the object from the S3 bucket
        response = s3_client.get_object(Bucket=bucket, Key=key)
        # Read the content of the file
        content = response['Body'].read().decode('utf-8')
        # Parse the content as JSON
        data = json.loads(content)
        return data
    except Exception as e:
        print(f"Error downloading or parsing JSON from S3: {e}")
        return None

# Function to filter entries by cutoff date
def _filter_entries_by_date(data, cutoff_datetime):
    filtered_data = []
    for entry in data:
        # Assuming the entry has a key 'upload_date' which contains the upload date as a string
        try:
            # Parse the upload date string into a datetime object
            entry_datetime = datetime.fromisoformat(entry['upload_date'])
            # If the entry's date is after the cutoff, add it to the filtered list
            if entry_datetime > cutoff_datetime:
                filtered_data.append(entry)
        except ValueError:
            print(f"Invalid date format in entry: {entry}")
    return filtered_data

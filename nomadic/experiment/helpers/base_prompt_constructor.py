from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pydantic import BaseModel
import re
import string

class BasePromptConstructor(ABC, BaseModel):
    """Abstract base class for prompt construction."""

    @abstractmethod
    def construct_prompt(self, template: str, params: Dict[str, Any], **kwargs) -> str:
        """Construct a prompt from a template and parameters."""
        pass

    @abstractmethod
    def extract_response(self, response: str, **kwargs) -> str:
        """Extract the relevant part of the response."""
        pass

    def validate_prompt(self, prompt: str) -> bool:
        """Validate the constructed prompt."""
        return bool(prompt and prompt.strip())

class DefaultPromptConstructor(BasePromptConstructor):
    """Default implementation of prompt construction with string formatting and response extraction."""

    def construct_prompt(self, template: str, params: Dict[str, Any], **kwargs) -> str:
        """
        Construct a prompt by formatting the template with provided parameters.

        Args:
            template: String template with placeholders
            params: Dictionary of parameter values
            **kwargs: Additional formatting options

        Returns:
            Formatted prompt string
        """
        try:
            # Handle example dictionary if provided
            example = kwargs.get('example', {})

            # Prepare parameters with fallbacks matching original implementation
            formatted_params = {
                'context': example.get('context', params.get('context', '')),
                'query': example.get('query', params.get('query', '')) or params.get('question', ''),
                'instruction': example.get('instruction', params.get('instruction', '')),
                'response': example.get('response', params.get('response', '')) or params.get('answer', '')
            }

            # Update with any additional parameters
            for key, value in params.items():
                if key not in formatted_params:
                    formatted_params[key] = value

            # Format the template with parameters
            prompt = template.format(**formatted_params)

            # Apply any additional formatting from kwargs
            if 'prefix' in kwargs:
                prompt = f"{kwargs['prefix']}\n{prompt}"
            if 'suffix' in kwargs:
                prompt = f"{prompt}\n{kwargs['suffix']}"

            return prompt.strip()

        except (KeyError, IndexError, ValueError) as e:
            print(f"Error constructing prompt: {str(e)}")
            return template  # Return original template on error

    def extract_response(self, response: str, **kwargs) -> str:
        """
        Extract relevant content from response using configurable patterns.

        Args:
            response: Raw response string
            **kwargs: Extraction options including:
                     - pattern: Regex pattern for extraction
                     - start_marker: Start of content marker
                     - end_marker: End of content marker

        Returns:
            Extracted response content
        """
        if not response:
            return ""

        try:
            # Try pattern-based extraction if provided
            if 'pattern' in kwargs:
                match = re.search(kwargs['pattern'], response)
                if match:
                    return match.group(1) if match.groups() else match.group(0)

            # Try marker-based extraction
            if 'start_marker' in kwargs and 'end_marker' in kwargs:
                start = response.find(kwargs['start_marker'])
                if start >= 0:
                    start += len(kwargs['start_marker'])
                    end = response.find(kwargs['end_marker'], start)
                    if end >= 0:
                        return response[start:end].strip()

            # Default to basic cleaning if no extraction method specified
            cleaned = response.strip()
            # Remove common formatting artifacts
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
            cleaned = re.sub(r'[^\w\s.,?!-]', '', cleaned)  # Remove special chars
            return cleaned

        except Exception as e:
            print(f"Error extracting response: {str(e)}")
            return response  # Return original response on error

"""
Private phrase extraction using the Document Privacy API.

This module provides the Tagger class for automatically identifying
sensitive/private phrases in documents using an external API.
"""

from typing import List

import requests


def find_phrase_offsets(text: str, phrases: List[str]) -> List[List[int]]:
    """
    Find all occurrences of phrases in text and return [start, end] offsets.

    Args:
        text: The full text to search in
        phrases: List of phrases to find

    Returns:
        List of [start_char, end_char] offsets for all phrase occurrences
    """
    offsets = []
    for phrase in phrases:
        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx == -1:
                break
            offsets.append([idx, idx + len(phrase)])
            start = idx + 1
    return offsets


class Tagger:
    """
    Private phrase extraction using the Document Privacy API.

    The Tagger uses an external API to identify sensitive information
    in documents. It supports different extraction models and document
    types (constitutions).

    Example:
        >>> tagger = Tagger(api_key="sk_...")
        >>> tagger.set_model("llama3.1-8b")
        >>> tagger.set_constitution("HEALTH")
        >>> phrases = tagger.extract_private_phrases("John Doe visited on 01/01/1990.")
        >>> print(phrases)
        ['John Doe', '01/01/1990']

    Args:
        api_key: API key for the Document Privacy API
        verbose: If True, log input/output of API calls (default: False)
    """

    def __init__(self, api_key: str, verbose: bool = False):
        """
        Initialize the Tagger with an API key.

        Args:
            api_key: API key for the Document Privacy API
            verbose: If True, log input/output of API calls (default: False)
        """
        self.api_key = api_key
        self.api_base = "https://api.documentprivacy.com"
        self._model = "llama3.1-8b"
        self._constitution = "HEALTH"
        self.verbose = verbose

    def set_model(self, model: str):
        """
        Set the extraction model.

        Args:
            model: Model identifier (e.g., 'llama3.1-8b')
        """
        self._model = model

    def set_constitution(self, constitution: str):
        """
        Set the document type/constitution.

        Available constitutions depend on the API. Common options:
        - 'HEALTH': Medical/healthcare documents
        - 'FINANCE': Financial documents
        - 'LEGAL': Legal documents

        Args:
            constitution: Document type identifier
        """
        self._constitution = constitution

    def get_available_models(self) -> List[str]:
        """
        Get list of available models from the API.

        Returns:
            List of available model identifiers

        Raises:
            requests.RequestException: If API call fails
        """
        url = f"{self.api_base}/models"
        headers = {
            "X-API-KEY": self.api_key
        }

        if self.verbose:
            print(f"[Tagger] GET {url}")

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        result = response.json()

        if self.verbose:
            print(f"[Tagger] Response: {result}")

        return result

    def extract_private_phrases(self, document: str) -> List[str]:
        """
        Extract private phrases from a document using the API.

        This method sends the document to the Document Privacy API,
        which uses the configured model and constitution to identify
        sensitive information.

        Args:
            document: The text document to analyze

        Returns:
            List of detected private/sensitive phrases

        Raises:
            requests.RequestException: If API call fails

        Example:
            >>> tagger = Tagger(api_key="sk_...")
            >>> phrases = tagger.extract_private_phrases(
            ...     "Patient John Smith, DOB 05/15/1980, was diagnosed with diabetes."
            ... )
            >>> print(phrases)
            ['John Smith', '05/15/1980', 'diabetes']
        """
        url = f"{self.api_base}/extract"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "document": document,
            "model": self._model,
            "type": self._constitution
        }

        if self.verbose:
            print(f"[Tagger] POST {url}")
            print(f"[Tagger] Input document: {document[:200]}{'...' if len(document) > 200 else ''}")
            print(f"[Tagger] Model: {self._model}, Constitution: {self._constitution}")

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        private_phrases = data.get("private_phrases", [])

        if self.verbose:
            print(f"[Tagger] Extracted phrases: {private_phrases}")

        return private_phrases

"""Tests for the Azure OpenAI Whisper parser."""

from unittest.mock import MagicMock, patch

import pytest

from langchain_community.document_loaders.parsers.audio import AzureOpenAIWhisperParser


@pytest.mark.requires("openai")
@patch("openai.AzureOpenAI")
def test_azure_openai_whisper(mock_client: MagicMock) -> None:
    endpoint = "endpoint"
    key = "key"
    name = "model"
    version = "37"

    parser = AzureOpenAIWhisperParser(
        api_key=key, azure_endpoint=endpoint, deployment_name=name, api_version=version
    )
    mock_client.assert_called_once_with(
        endpoint=endpoint, key=key, deployment_name=name, version=version
    )
    assert parser._client == mock_client()

"""Tests for the Azure OpenAI Whisper parser."""

import io
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents.base import Blob

from langchain_community.document_loaders.parsers.audio import AzureOpenAIWhisperParser


@pytest.mark.requires("openai")
@patch("openai.AzureOpenAI")
def test_azure_openai_whisper(mock_client: MagicMock) -> None:
    endpoint = "endpoint"
    key = "key"
    version = "115"
    name = "model"

    parser = AzureOpenAIWhisperParser(
        api_key=key, azure_endpoint=endpoint, api_version=version, deployment_name=name
    )
    mock_client.assert_called_once_with(
        api_key=key,
        azure_endpoint=endpoint,
        api_version=version,
        max_retries=3,
        azure_ad_token=None,
    )
    assert parser._client == mock_client()


@pytest.mark.requires("openai")
@patch("openai.AzureOpenAI")
@patch("openai.AzureOpenAI.audio.transcriptions.create")
def test_azure_openai_whisper_lazy_parse(
    mock_transcribe: MagicMock, mock_client: MagicMock
) -> None:
    endpoint = "endpoint"
    key = "key"
    version = "115"
    name = "model"

    mock_client.return_value = mock_client

    parser = AzureOpenAIWhisperParser(
        api_key=key, azure_endpoint=endpoint, api_version=version, deployment_name=name
    )

    mock_client.assert_called_once_with(
        api_key=key,
        azure_endpoint=endpoint,
        api_version=version,
        max_retries=3,
        azure_ad_token=None,
    )

    mock_response = MagicMock()
    mock_response.text = "This is a mock transcription"
    mock_transcribe.return_value = mock_response

    blob = Blob(path="audio_path.m4a", data=b"Great day for fishing ain't it")
    docs = parser.lazy_parse(blob=blob)

    file_obj = io.BytesIO(b"Great day for fishing ain't it")
    mock_transcribe.assert_called_once_with(
        model=name,
        file=file_obj,
    )
    for doc in docs:
        assert doc.page_content == mock_response.text

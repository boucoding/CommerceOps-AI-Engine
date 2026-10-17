from types import SimpleNamespace

import openai
import pytest

from src.config import LLMMode
from src.llm import build_llm_client
from src.llm.base import LLMError, ReplayMissError
from src.llm.openai_client import OpenAILLMClient
from src.llm.replay import RecordingLLMClient, ReplayLLMClient
from src.models.llm_schemas import IntentClassification
from tests.conftest import ScriptedLLM, intent, make_settings

INTENT = intent("order request", "Liz")


def _completion(parsed=None, content=None, refusal=None):
    message = SimpleNamespace(parsed=parsed, content=content, refusal=refusal)
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class StubOpenAI:
    def __init__(self, response=None, error=None, embedding_dim=4):
        self.kwargs = []
        completions = SimpleNamespace(parse=self._call(response, error), create=self._call(response, error))
        self.chat = SimpleNamespace(completions=completions)
        self.embeddings = SimpleNamespace(create=self._embed(embedding_dim))

    def _call(self, response, error):
        def call(**kwargs):
            self.kwargs.append(kwargs)
            if error:
                raise error
            return response

        return call

    def _embed(self, dim):
        def create(model, input):
            data = [SimpleNamespace(index=i, embedding=[float(i)] * dim) for i in reversed(range(len(input)))]
            return SimpleNamespace(data=data)

        return create


def test_openai_parse_returns_parsed_and_usage():
    parsed = IntentClassification.model_validate(INTENT)
    stub = StubOpenAI(_completion(parsed=parsed))
    client = OpenAILLMClient(make_settings(LLM_MODEL="gpt-6-luna"), client=stub)  # type: ignore[arg-type]
    result = client.parse(task="classify", system="s", user="u", schema=IntentClassification)
    assert result.value.customer_name == "Liz"
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (11, 7)
    assert stub.kwargs[0]["model"] == "gpt-6-luna" and "temperature" not in stub.kwargs[0]


def test_temperature_is_sent_only_when_configured():
    stub = StubOpenAI(_completion(content="hi"))
    OpenAILLMClient(make_settings(LLM_TEMPERATURE=0.2), client=stub).complete(task="t", system="s", user="u")  # type: ignore[arg-type]
    assert stub.kwargs[0]["temperature"] == 0.2


@pytest.mark.parametrize(
    "response, error",
    [
        (_completion(refusal="no"), None),
        (_completion(parsed=None), None),
        (None, openai.APIConnectionError(request=SimpleNamespace())),  # type: ignore[arg-type]
    ],
)
def test_openai_failures_become_llm_errors(response, error):
    client = OpenAILLMClient(make_settings(), client=StubOpenAI(response, error))  # type: ignore[arg-type]
    with pytest.raises(LLMError):
        client.parse(task="classify", system="s", user="u", schema=IntentClassification)


def test_embeddings_are_ordered_and_dimension_checked():
    client = OpenAILLMClient(make_settings(EMBEDDING_DIMENSIONS=4), client=StubOpenAI())  # type: ignore[arg-type]
    assert client.embed(["a", "b"]) == [[0.0] * 4, [1.0] * 4]
    bad = OpenAILLMClient(make_settings(EMBEDDING_DIMENSIONS=8), client=StubOpenAI())  # type: ignore[arg-type]
    with pytest.raises(LLMError, match="dimensions"):
        bad.embed(["a"])


def test_record_then_replay_roundtrip(tmp_path):
    path = str(tmp_path / "replays.json")
    scripted = ScriptedLLM({"classify": INTENT, "reply": "Hello there"})
    scripted.embedding_model = "openai:test-embed"
    recorder = RecordingLLMClient(scripted, path)
    recorder.parse(task="classify", system="s", user="Email ID: E7\nbody", schema=IntentClassification)
    recorder.complete(task="reply", system="s", user="Email ID: E7\nbody")
    vectors = recorder.embed(["catalog text"])
    assert recorder.save() == 2

    replay = ReplayLLMClient(path, dimensions=1536)
    assert replay.source == "recorded" and replay.model == "scripted-model"
    result = replay.parse(task="classify", system="s", user="Email ID: E7\nbody", schema=IntentClassification)
    assert result.replayed and result.value.customer_name == "Liz"
    assert replay.complete(task="reply", system="s", user="Email ID: E7\nbody").value == "Hello there"
    assert replay.embed(["catalog text"])[0] == pytest.approx(vectors[0], abs=1e-6)

    with pytest.raises(ReplayMissError):
        replay.complete(task="reply", system="s (edited prompt)", user="Email ID: E7\nbody")
    with pytest.raises(ReplayMissError):
        replay.embed(["never embedded"])


def test_factory_modes(tmp_path):
    assert build_llm_client(make_settings(), LLMMode.OFFLINE) is None
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_settings(LLM_MODE=LLMMode.LIVE, OPENAI_API_KEY="your_openai_api_key_here").resolved_llm_mode()
    missing = make_settings(LLM_MODE=LLMMode.AUTO, REPLAY_FIXTURES_PATH=str(tmp_path / "none.json"))
    assert missing.resolved_llm_mode() == LLMMode.OFFLINE
    assert make_settings(LLM_MODE=LLMMode.AUTO, OPENAI_API_KEY="sk-real").resolved_llm_mode() == LLMMode.LIVE

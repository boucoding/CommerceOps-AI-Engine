import os
from types import SimpleNamespace

import openai
import pytest

from src.agent.schemas import Plan, SearchCatalogArgs
from src.agent.tools import Tool
from src.config import LLMMode, Settings
from src.llm import build_llm_client
from src.llm.base import LLMError, ReplayMissError
from src.llm.openai_client import OpenAILLMClient
from src.llm.replay import RecordingLLMClient, ReplayLLMClient
from tests.conftest import ScriptedLLM, make_settings, plan

PLAN = plan("order request", "search_catalog", "submit_reply", name="Liz")


def _completion(parsed=None, content=None, refusal=None, tool_calls=None):
    message = SimpleNamespace(parsed=parsed, content=content, refusal=refusal, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7)
    )


class StubOpenAI:
    def __init__(self, response=None, error=None, dim=4):
        self.kwargs = []

        def call(**kwargs):
            self.kwargs.append(kwargs)
            if error:
                raise error
            return response

        def embed(model, input):
            return SimpleNamespace(
                data=[SimpleNamespace(index=i, embedding=[float(i)] * dim) for i in reversed(range(len(input)))]
            )

        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=call, create=call))
        self.embeddings = SimpleNamespace(create=embed)


def client(stub, **settings):
    return OpenAILLMClient(make_settings(**settings), client=stub)  # type: ignore[arg-type]


def test_parse_returns_structured_plan():
    stub = StubOpenAI(_completion(parsed=Plan.model_validate(PLAN)))
    result = client(stub, LLM_MODEL="gpt-6-luna").parse(task="plan", system="s", user="u", schema=Plan)
    assert result.value.customer_name == "Liz" and result.usage.prompt_tokens == 11
    assert stub.kwargs[0]["model"] == "gpt-6-luna" and "temperature" not in stub.kwargs[0]


def test_chat_requires_a_single_tool_call_per_turn():
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="search_catalog", arguments='{"query": "x"}'))
    stub = StubOpenAI(_completion(tool_calls=[call]))
    turn = client(stub, LLM_TEMPERATURE=0.1).chat(task="execute", messages=[], tools=[]).value
    assert turn.tool_calls[0].name == "search_catalog"
    assert stub.kwargs[0]["tool_choice"] == "required" and stub.kwargs[0]["parallel_tool_calls"] is False
    assert stub.kwargs[0]["temperature"] == 0.1


@pytest.mark.parametrize(
    "response, error",
    [
        (_completion(refusal="no"), None),
        (_completion(parsed=None), None),
        (None, openai.APIConnectionError(request=SimpleNamespace())),  # type: ignore[arg-type]
    ],
)
def test_failures_become_llm_errors(response, error):
    with pytest.raises(LLMError):
        client(StubOpenAI(response, error)).parse(task="plan", system="s", user="u", schema=Plan)


def test_embeddings_ordered_and_dimension_checked():
    assert client(StubOpenAI(), EMBEDDING_DIMENSIONS=4).embed(["a", "b"]) == [[0.0] * 4, [1.0] * 4]
    with pytest.raises(LLMError, match="dimensions"):
        client(StubOpenAI(), EMBEDDING_DIMENSIONS=8).embed(["a"])


def test_tool_schemas_are_strict():
    schema = Tool("search_catalog", "d", SearchCatalogArgs, lambda a, c: {}).schema()
    fn = schema["function"]
    assert fn["strict"] is True and fn["parameters"]["additionalProperties"] is False
    assert fn["parameters"]["required"] == ["query"]


def test_record_then_replay_roundtrip(tmp_path):
    path = str(tmp_path / "rec.json")
    scripted = ScriptedLLM(plans={"E7": PLAN}, steps={"E7": [("search_catalog", {"query": "boots"})]})
    recorder = RecordingLLMClient(scripted, path)
    recorder.parse(task="plan", system="s", user="Email ID: E7\nbody", schema=Plan)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "Email ID: E7\nbody"}]
    recorded_turn = recorder.chat(task="execute", messages=messages, tools=[]).value
    vector = recorder.embed(["catalog text"])[0]
    assert recorder.save() == 2 and os.stat(path).st_mode & 0o777 == 0o644

    replay = ReplayLLMClient(path)
    assert replay.parse(task="plan", system="s", user="Email ID: E7\nbody", schema=Plan).replayed
    assert replay.chat(task="execute", messages=messages, tools=[]).value == recorded_turn
    assert replay.embed(["catalog text"])[0] == pytest.approx(vector, abs=1e-6)
    with pytest.raises(ReplayMissError):
        replay.parse(task="plan", system="edited prompt", user="Email ID: E7\nbody", schema=Plan)
    with pytest.raises(ReplayMissError):
        replay.embed(["never embedded"])


def test_mode_validation(tmp_path):
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_llm_client(make_settings(LLM_MODE=LLMMode.LIVE, OPENAI_API_KEY="your_openai_api_key_here"))
    with pytest.raises(FileNotFoundError, match="make record"):
        build_llm_client(make_settings(LLM_MODE=LLMMode.REPLAY, REPLAY_FIXTURES_PATH=str(tmp_path / "x.json")))


def test_env_example_loads_with_blank_values(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=".env.example")  # type: ignore[call-arg]
    assert settings.LLM_MODEL == "gpt-6-luna" and settings.LLM_TEMPERATURE is None
    assert settings.OPENAI_BASE_URL is None and settings.API_KEY is None and not settings.has_openai_key

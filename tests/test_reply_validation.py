import pytest

from src.safety.reply_validation import ReplyContext, ReplyValidationError, cited_product_ids, finalize_reply
from tests.conftest import make_settings

SETTINGS = make_settings()


def test_valid_reply_gets_signature_and_restored_pii():
    ctx = ReplyContext({"SWL2345"}, {"SWL2345"}, {"{{EMAIL_1}}": "j@x.com"})
    text = finalize_reply("Hi Jo,\n\nYour Sleek Wallet (SWL2345) ships today; receipt to {{EMAIL_1}}.", ctx, SETTINGS)
    assert "j@x.com" in text and text.endswith(SETTINGS.signature_block)


@pytest.mark.parametrize(
    "body, error",
    [
        ("Thanks, [Your Name]", "placeholder"),
        ("Try the Leather Tote (LTH5432).", "no tool returned"),
        ("Your order is confirmed.", "every reserved product"),
        ("We'll call you at {{PHONE_9}}.", "redaction"),
        ("   ", "empty"),
        ("x" * 5000, "longer than"),
    ],
)
def test_rejections(body, error):
    ctx = ReplyContext({"SWL2345"}, {"SWL2345"} if error == "every reserved product" else set(), {})
    with pytest.raises(ReplyValidationError, match=error):
        finalize_reply(body, ctx, SETTINGS)


def test_product_id_detection_is_format_agnostic():
    assert cited_product_ids("CBT8901, SKU-1042A, AB12 but not 1950s, E009 or Hello") == ["AB12", "CBT8901", "SKU1042A"]

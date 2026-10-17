import pytest

from src.models.domain import STATUS_CREATED, STATUS_OUT_OF_STOCK, OrderLine, Product
from src.pipeline import template_engine as tpl
from tests.conftest import make_settings

SETTINGS = make_settings()


def ctx(**kw):
    defaults = {"language": "en", "customer_name": "Jessica", "allowed_product_ids": {"SWL2345"}}
    defaults.update(kw)
    return tpl.ReplyContext(**defaults)


def test_reply_is_framed_with_greeting_and_brand_signature():
    text = tpl.finalize_reply("Your Sleek Wallet (SWL2345) is on its way.", ctx(), SETTINGS)
    assert text.startswith("Dear Jessica,\n\n")
    assert text.endswith(SETTINGS.signature_block)
    assert "Warm regards," in text


def test_model_greeting_and_signoff_are_replaced():
    body = "Dear Customer,\n\nYour Sleek Wallet (SWL2345) ships today.\n\nBest regards,\nAura Team"
    text = tpl.finalize_reply(body, ctx(customer_name=None), SETTINGS)
    assert text.count("Dear") == 0 and text.startswith("Hello,")
    assert "Aura Team" not in text and "ships today." in text


def test_spanish_framing():
    text = tpl.finalize_reply("Está disponible.", ctx(language="es", customer_name=None), SETTINGS)
    assert text.startswith("Hola,") and "Saludos cordiales," in text


@pytest.mark.parametrize(
    "body, error",
    [
        ("Thanks, [Your Name]", "placeholder"),
        ("You might like the Leather Tote (LTH5432).", "ungrounded"),
        ("Your order is confirmed.", "omits ordered"),
        ("We'll call you at {{PHONE_9}}.", "redaction"),
        ("", "empty"),
    ],
)
def test_validation_rejects_unsafe_replies(body, error):
    context = ctx(required_product_ids={"SWL2345"} if error == "omits ordered" else (), token_map={})
    with pytest.raises(tpl.ReplyValidationError, match=error):
        tpl.finalize_reply(body, context, SETTINGS)


def test_pii_tokens_are_restored():
    text = tpl.finalize_reply("We'll email {{EMAIL_1}}.", ctx(token_map={"{{EMAIL_1}}": "j@x.com"}), SETTINGS)
    assert "j@x.com" in text


def test_order_template_states_facts():
    lines = [
        OrderLine("SWL2345", "Sleek Wallet", 1, 1, STATUS_CREATED, 4),
        OrderLine("RSG8901", "Retro Sunglasses", 2, 0, STATUS_OUT_OF_STOCK, 1),
    ]
    alt = Product("BKR0123", "Bucket Hat", "Accessories", "", 3, "Summer", 39.99)
    body = tpl.order_template(lines, {"RSG8901": [alt]}, SETTINGS)
    assert "Sleek Wallet (SWL2345) x 1" in body
    assert "2 x Retro Sunglasses (RSG8901)" in body and "only 1 in stock" in body
    assert "Bucket Hat (BKR0123, $39.99)" in body
    final = tpl.finalize_reply(body, ctx(allowed_product_ids={"SWL2345", "RSG8901", "BKR0123"}), SETTINGS)
    assert "[" not in final


def test_clarification_and_escalation_templates_pass_validation():
    for body in (tpl.clarification_template(SETTINGS), tpl.escalation_template(SETTINGS)):
        assert tpl.finalize_reply(body, ctx(allowed_product_ids=set()), SETTINGS)

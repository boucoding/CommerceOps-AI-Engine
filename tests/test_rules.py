import pytest

from src.models.domain import ORDER_REQUEST, PRODUCT_INQUIRY
from src.pipeline import rules


@pytest.mark.parametrize(
    "body, expected",
    [
        ("I want to order 2 wallets.", ORDER_REQUEST),
        ("Please send me the green scarf.", ORDER_REQUEST),
        ("I'm ready to place my order for the boots.", ORDER_REQUEST),
        ("I was thinking of ordering the boots, but I'll wait until Fall.", PRODUCT_INQUIRY),
        ("What material is the beanie made of?", PRODUCT_INQUIRY),
        ("I bought a wallet last year and love it. Do you have a matching belt?", PRODUCT_INQUIRY),
        ("I also need to get my lawnmower fixed. Which bags do you have?", PRODUCT_INQUIRY),
        ("I'd like to buy the boots! I'd like to order sunglasses too, but probably next time.", ORDER_REQUEST),
    ],
)
def test_classify_intent(body, expected):
    assert rules.classify_intent("", body) == expected


@pytest.mark.parametrize(
    "body, name",
    [
        ("Hi, my name is Claire and I need a dress.", "Claire"),
        ("Hi, this is Liz. Please send me 5 beanies.", "Liz"),
        ("Please send me 1 Sleek Wallet. Thanks, Johny", "Johny"),
        ("Is it good for spring? Thank you, Antonio", "Antonio"),
        # Regressions: the old extractor returned "Your", "In" and "You" for these.
        ("My wife Emily got me a briefcase from your store.", None),
        ("I'm looking for a bag for my husband Thomas. Thank you in advance!", None),
        ("I would like to buy boots from you guys!", None),
    ],
)
def test_extract_customer_name(body, name):
    assert rules.extract_customer_name(body) == name


def test_name_must_appear_in_text():
    assert rules.is_plausible_name("Sam", "My name is Sam")
    assert not rules.is_plausible_name("Samantha", "My name is Sam")
    assert not rules.is_plausible_name("Your", "from Your store")


@pytest.mark.parametrize(
    "segment, quantity, all_available",
    [
        ("please send me 5 ", 5, False),
        ("I'd like to order three to four ", 4, False),
        ("I want to order all the remaining ", None, True),
        (" Cargo Pants - all 5 of them", 5, False),
        ("buy 2 pairs of ", 2, False),
        ("order a pair of ", 1, False),
        ("it costs $22 and ", None, False),
    ],
)
def test_parse_quantity(segment, quantity, all_available):
    parsed = rules.parse_quantity(segment)
    if quantity is None and not all_available:
        assert parsed is None
    else:
        assert parsed is not None
        assert (parsed.quantity, parsed.wants_all_available) == (quantity, all_available)


def test_detect_language():
    assert rules.detect_language("Hola, tengo una pregunta. ¿De qué material está hecho?") == "es"
    assert rules.detect_language("Hello there") == "en"

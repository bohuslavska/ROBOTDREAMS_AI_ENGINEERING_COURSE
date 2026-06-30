"""
Synthetic dataset generator for customer support email extraction.

Produces (email_text, expected_json) pairs deterministically — no LLM API calls.
The JSON schema:
    {
      "customer_name": str | null,
      "product": str,
      "issue_category": "billing|technical|account|feature_request|other",
      "urgency": "low|medium|high|critical",
      "summary": str
    }
"""

import json
import random
import hashlib
from pathlib import Path

random.seed(42)

FIRST_NAMES = [
    "Oleksii", "Maria", "Ivan", "Anna", "Petro", "Olha", "Andriy", "Kateryna",
    "Dmytro", "Iryna", "Sergii", "Natalia", "Volodymyr", "Yulia", "Mykhailo",
    "Tetyana", "Bohdan", "Svitlana", "Roman", "Vira", "Yaroslav", "Halyna",
    "Mark", "Sarah", "James", "Emily", "Robert", "Jessica", "Michael", "Ashley",
    "David", "Amanda", "Daniel", "Jennifer", "Matthew", "Lisa", "Christopher",
    "Karen", "Joseph", "Nancy", "John", "Patricia", "Thomas", "Linda",
]
LAST_NAMES = [
    "Shevchenko", "Bondarenko", "Tkachenko", "Kovalenko", "Melnyk", "Boyko",
    "Pavlenko", "Marchenko", "Lysenko", "Klymenko", "Smith", "Johnson",
    "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Wilson",
    "Anderson", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson",
]
PRODUCTS = [
    "Free Plan", "Starter Plan", "Pro Plan", "Premium Plan", "Enterprise Plan",
    "Mobile App", "Desktop App", "Web Dashboard", "API Access", "Analytics Add-on",
    "Team Seats", "SSO Add-on", "Priority Support",
]

CATEGORIES = ["billing", "technical", "account", "feature_request", "other"]
URGENCIES = ["low", "medium", "high", "critical"]


# ─────────────────────────────────────────────────────────────────────────────
# Email templates — each produces (email_text, expected_json) for ONE category
# ─────────────────────────────────────────────────────────────────────────────

def gen_billing(rng, name, product, urgency_hint):
    templates = [
        ("Hi, I was charged twice for {product} this month. My card shows two charges of $29.99 on the same day. Please refund the duplicate. — {name}",
         "Duplicate charge for {product}, refund requested"),
        ("Hello support team, I cancelled my {product} subscription last week but I just got billed again today. This is unacceptable. Please refund immediately and confirm cancellation. Best, {name}",
         "Billed after cancellation, refund and cancellation confirmation requested"),
        ("Hey, can you explain the line item 'overage fee' on my last invoice for {product}? I don't understand what it is. Thanks, {name}",
         "Confusion about overage fee on invoice"),
        ("My company needs an invoice with our VAT number for {product}. Can you re-issue last month's invoice? — {name}",
         "Re-issued invoice with VAT number requested"),
        ("Hi, I want to upgrade from {product} to a higher tier but pay annually. What discount do I get? — {name}",
         "Annual upgrade pricing question"),
        ("URGENT: my payment failed and {product} got suspended. We need it back NOW, client meeting in 1 hour!! — {name}",
         "Payment failure, account suspended, urgent reactivation needed"),
    ]
    tpl_email, tpl_summary = rng.choice(templates)
    email = tpl_email.format(product=product, name=name)
    summary = tpl_summary.format(product=product)
    return email, "billing", summary


def gen_technical(rng, name, product, urgency_hint):
    templates = [
        ("Hi, {product} keeps crashing when I try to export to CSV. I've tried restarting 3 times. Logs attached. — {name}",
         "{product} crashes on CSV export"),
        ("The {product} integration with Slack stopped working yesterday around 3pm. No errors, just no messages get through. {name}",
         "{product} Slack integration silently broken"),
        ("API endpoint /v1/users returns 500 for {product} requests. Was working fine yesterday. — {name}",
         "{product} API /v1/users returning 500"),
        ("Login broken on {product} mobile. After entering password I get 'something went wrong'. iOS 17. — {name}",
         "{product} mobile login error on iOS 17"),
        ("Hey, sync between {product} desktop and web takes 10+ minutes now. Used to be instant. — {name}",
         "{product} sync latency degraded"),
        ("CRITICAL: production data missing in {product}! Filter shows 0 results but we have 50k records. URGENT! — {name}",
         "{product} showing 0 records, possible data loss"),
        ("Webhook deliveries for {product} are failing with timeout errors. Retry logic broken? — {name}",
         "{product} webhook delivery timeouts"),
    ]
    tpl_email, tpl_summary = rng.choice(templates)
    email = tpl_email.format(product=product, name=name)
    summary = tpl_summary.format(product=product)
    return email, "technical", summary


def gen_account(rng, name, product, urgency_hint):
    templates = [
        ("Hi, I forgot the email I used to sign up for {product}. How can I recover access? — {name}",
         "Account recovery, forgot signup email"),
        ("Need to add 3 more seats to our {product} team account. How? Thanks, {name}",
         "Add seats to {product} team"),
        ("Please delete my account and all data per GDPR. Email: x@y.com. — {name}",
         "GDPR account deletion request"),
        ("How do I transfer ownership of our {product} workspace to a colleague? They will take over. — {name}",
         "Workspace ownership transfer question"),
        ("Two-factor auth is broken on {product}. Lost phone, codes don't work. Need backup access. — {name}",
         "2FA recovery needed"),
        ("Can you merge two {product} accounts I have? Same person, two emails. — {name}",
         "Account merge request"),
        ("Update billing email for {product} from old@x.com to new@x.com please. — {name}",
         "Billing email update"),
    ]
    tpl_email, tpl_summary = rng.choice(templates)
    email = tpl_email.format(product=product, name=name)
    summary = tpl_summary.format(product=product)
    return email, "account", summary


def gen_feature_request(rng, name, product, urgency_hint):
    templates = [
        ("Would be great if {product} supported dark mode. Eyes hurt at night. — {name}",
         "Dark mode feature request"),
        ("Pls add bulk export to {product}. Need to download 1000+ records monthly. — {name}",
         "Bulk export feature request"),
        ("Hi, can {product} integrate with Linear? We use it for our roadmap. — {name}",
         "Linear integration request"),
        ("Suggestion: {product} should let users schedule recurring reports. Currently we run them manually. — {name}",
         "Recurring scheduled reports request"),
        ("Please add keyboard shortcuts in {product} for power users. Mouse-only is slow. — {name}",
         "Keyboard shortcuts request"),
        ("It would be amazing if {product} had a public API for the analytics dashboard. — {name}",
         "Public API for analytics dashboard"),
        ("Wishlist: webhook signatures in {product} (HMAC). Security audit asking. — {name}",
         "HMAC webhook signature request"),
    ]
    tpl_email, tpl_summary = rng.choice(templates)
    email = tpl_email.format(product=product, name=name)
    summary = tpl_summary.format(product=product)
    return email, "feature_request", summary


def gen_other(rng, name, product, urgency_hint):
    templates = [
        ("Hi, where can I find documentation for {product}? — {name}",
         "Documentation link request"),
        ("Do you have a referral program for {product}? Interested. — {name}",
         "Referral program inquiry"),
        ("Is {product} HIPAA compliant? Need for healthcare client. — {name}",
         "HIPAA compliance question"),
        ("Looking for a {product} case study in fintech. Any available? — {name}",
         "Fintech case study request"),
        ("Hi team, just wanted to say {product} is awesome. Keep it up! — {name}",
         "Positive feedback / testimonial"),
        ("Is there a status page for {product}? Want to monitor uptime. — {name}",
         "Status page link request"),
        ("Are you hiring? I love {product} and want to join the team. — {name}",
         "Job inquiry"),
    ]
    tpl_email, tpl_summary = rng.choice(templates)
    email = tpl_email.format(product=product, name=name)
    summary = tpl_summary.format(product=product)
    return email, "other", summary


GENERATORS = {
    "billing": gen_billing,
    "technical": gen_technical,
    "account": gen_account,
    "feature_request": gen_feature_request,
    "other": gen_other,
}


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases for eval set
# ─────────────────────────────────────────────────────────────────────────────

EDGE_CASES = [
    # No customer name (signed only with initials or anonymous)
    {
        "email": "Hi, can you tell me when the next maintenance window for Pro Plan is? — A.",
        "expected": {
            "customer_name": None,
            "product": "Pro Plan",
            "issue_category": "other",
            "urgency": "low",
            "summary": "Maintenance window schedule inquiry",
        },
    },
    # Multiple issues — pick primary by urgency
    {
        "email": "Hello, two things: (1) my Pro Plan was double charged this month, (2) also dark mode would be nice. Refund is priority. — Maria Bondarenko",
        "expected": {
            "customer_name": "Maria Bondarenko",
            "product": "Pro Plan",
            "issue_category": "billing",
            "urgency": "high",
            "summary": "Duplicate Pro Plan charge with secondary dark mode feature request",
        },
    },
    # Implicit urgency — critical because of production
    {
        "email": "Production is down. API Access returns 500 on every call. Customers complaining. — DevOps team @ Acme",
        "expected": {
            "customer_name": None,
            "product": "API Access",
            "issue_category": "technical",
            "urgency": "critical",
            "summary": "Production down, API returning 500",
        },
    },
    # Sarcastic / passive aggressive
    {
        "email": "Oh great, my Premium Plan got cancelled AGAIN without notice. Third time this year. Wonderful service. — Robert Johnson",
        "expected": {
            "customer_name": "Robert Johnson",
            "product": "Premium Plan",
            "issue_category": "billing",
            "urgency": "high",
            "summary": "Recurring unexpected cancellation of Premium Plan",
        },
    },
    # Vague product (mentions company but not specific tier)
    {
        "email": "Hi, your software keeps freezing on my laptop. Worked fine before update. — David Kim",
        "expected": {
            "customer_name": "David Kim",
            "product": "Desktop App",
            "issue_category": "technical",
            "urgency": "medium",
            "summary": "Desktop app freezing after update",
        },
    },
    # Non-English mixed in
    {
        "email": "Привіт! I cannot login to Mobile App. Password is correct. Дякую. — Iryna",
        "expected": {
            "customer_name": "Iryna",
            "product": "Mobile App",
            "issue_category": "technical",
            "urgency": "medium",
            "summary": "Mobile App login failure with correct password",
        },
    },
    # Question that looks technical but is actually billing
    {
        "email": "Why is my Pro Plan slower than my friend's Pro Plan? Are different tiers throttled? — Sarah Lee",
        "expected": {
            "customer_name": "Sarah Lee",
            "product": "Pro Plan",
            "issue_category": "billing",
            "urgency": "low",
            "summary": "Question about throttling differences between Pro Plan accounts",
        },
    },
    # Very short email
    {
        "email": "Cancel my account please. Premium Plan.",
        "expected": {
            "customer_name": None,
            "product": "Premium Plan",
            "issue_category": "account",
            "urgency": "medium",
            "summary": "Account cancellation request for Premium Plan",
        },
    },
    # Praise + question (mixed sentiment)
    {
        "email": "Love Analytics Add-on! Quick question — can it be customized per team? Thanks — Bohdan",
        "expected": {
            "customer_name": "Bohdan",
            "product": "Analytics Add-on",
            "issue_category": "feature_request",
            "urgency": "low",
            "summary": "Per-team customization for Analytics Add-on",
        },
    },
    # Phishing-looking email (still legit support request)
    {
        "email": "URGENT URGENT URGENT my Enterprise Plan account locked, send help, all caps because EMERGENCY!!! — JOHN SMITH",
        "expected": {
            "customer_name": "JOHN SMITH",
            "product": "Enterprise Plan",
            "issue_category": "account",
            "urgency": "critical",
            "summary": "Enterprise Plan account locked, urgent reactivation",
        },
    },
]


def determine_urgency(category, text):
    text_lower = text.lower()
    if any(w in text_lower for w in ["urgent", "critical", "production", "now", "asap", "emergency", "down"]):
        return "critical" if any(w in text_lower for w in ["critical", "production", "down", "emergency"]) else "high"
    if any(w in text_lower for w in ["unacceptable", "refund immediately", "please refund"]):
        return "high"
    if category == "feature_request" or category == "other":
        return "low" if "wishlist" in text_lower or "would be" in text_lower or "suggestion" in text_lower else "low"
    if category == "billing":
        return "high" if any(w in text_lower for w in ["refund", "double", "duplicate", "wrong"]) else "medium"
    return "medium"


def generate_example(rng, category=None):
    name_full = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    # Sometimes drop last name
    if rng.random() < 0.2:
        name_full = name_full.split()[0]
    # Sometimes anonymous
    if rng.random() < 0.05:
        name_full = None
    product = rng.choice(PRODUCTS)
    if category is None:
        category = rng.choice(CATEGORIES)

    gen = GENERATORS[category]
    email, cat, summary = gen(rng, name_full or "Anonymous", product, None)

    if name_full is None:
        # Strip the "— Anonymous" suffix
        email = email.replace("— Anonymous", "").rstrip(" -—")

    urgency = determine_urgency(cat, email)

    return {
        "email": email,
        "expected": {
            "customer_name": name_full,
            "product": product,
            "issue_category": cat,
            "urgency": urgency,
            "summary": summary,
        },
    }


def hash_email(email):
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()[:16]


def main():
    rng = random.Random(42)

    # ── Generate eval set: 10 edge cases + 20 standard (4 per category) ──
    eval_examples = list(EDGE_CASES)  # 10 pre-defined edge cases
    for cat in CATEGORIES:
        for _ in range(4):
            eval_examples.append(generate_example(rng, category=cat))

    # ── Generate training set: 300 examples, balanced across categories ──
    rng_train = random.Random(123)
    train_examples = []
    per_cat = 60  # 5 categories × 60 = 300
    for cat in CATEGORIES:
        for _ in range(per_cat):
            train_examples.append(generate_example(rng_train, category=cat))

    # ── Dedup vs eval (hash-based) ──
    eval_hashes = {hash_email(ex["email"]) for ex in eval_examples}
    train_examples = [ex for ex in train_examples if hash_email(ex["email"]) not in eval_hashes]
    print(f"Training examples after dedup: {len(train_examples)}")

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Write eval.jsonl ──
    with open(data_dir / "eval.jsonl", "w") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(eval_examples)} eval examples → {data_dir / 'eval.jsonl'}")

    # ── Write train.jsonl in OpenAI chat format ──
    SYSTEM_PROMPT = (
        "You extract structured data from customer support emails. "
        "Return only a single valid JSON object with fields: "
        "customer_name (string or null), product (string), "
        "issue_category (one of: billing, technical, account, feature_request, other), "
        "urgency (one of: low, medium, high, critical), "
        "summary (one short sentence). No extra text."
    )

    with open(data_dir / "train.jsonl", "w") as f:
        for ex in train_examples:
            row = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ex["email"]},
                    {"role": "assistant", "content": json.dumps(ex["expected"], ensure_ascii=False)},
                ]
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train_examples)} training examples → {data_dir / 'train.jsonl'}")

    # ── Print category distribution ──
    from collections import Counter
    eval_cats = Counter(ex["expected"]["issue_category"] for ex in eval_examples)
    train_cats = Counter(ex["expected"]["issue_category"] for ex in train_examples)
    print(f"\nEval distribution: {dict(eval_cats)}")
    print(f"Train distribution: {dict(train_cats)}")


if __name__ == "__main__":
    main()

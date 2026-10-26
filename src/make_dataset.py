"""Generate a small SYNTHETIC labelled dataset of support messages.

No real customer data is used. Messages are composed from hand-written
fragments (opener + core complaint + detail + closer) with random casing,
typos and punctuation noise, then de-duplicated and split with a stratified
train/test split (80/20) BEFORE any model sees them.

A separate hand-written "challenge" set (data/challenge.csv) is NOT generated
from these fragments and is used as a harder, out-of-template check.

Usage: python -m src.make_dataset
"""
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CORES = {
    "billing": [
        "I was charged twice for my subscription",
        "my card was billed two times this month",
        "I see a duplicate charge on my statement",
        "I need a refund for last month's payment",
        "why was I charged {amt} when my plan costs less",
        "the invoice amount is wrong",
        "please send me a copy of my invoice",
        "my payment failed but the money left my bank account",
        "I was billed after I cancelled my subscription",
        "I want to switch from monthly to yearly billing",
        "the coupon code was not applied at checkout",
        "I was overcharged on my latest bill",
        "can I get a receipt for my last payment",
        "my credit card keeps getting declined when I try to pay",
        "I did not authorise this {amt} charge",
        "how do I update my payment method",
        "the price on my renewal is higher than what I agreed to",
        "I need to change the billing address on my invoice",
        "the free trial ended and I was charged without a reminder",
        "tax was added to my invoice and I am exempt",
        "my refund has not arrived after {n} days",
        "there is an unknown transaction from your company on my card",
    ],
    "technical": [
        "the app crashes every time I open it",
        "the dashboard will not load and shows a blank page",
        "I keep getting a 500 error when I upload a file",
        "the export to CSV button does nothing",
        "the website is extremely slow today",
        "notifications are not working on my phone",
        "the mobile app freezes when I tap settings",
        "I get a timeout error when syncing my data",
        "the API returns a 403 even with a valid token",
        "images are not loading in the editor",
        "the page shows an error code {code}",
        "search stopped returning any results",
        "the integration with Slack stopped working",
        "my files fail to upload and the progress bar is stuck",
        "the video player keeps buffering and then stops",
        "the app is not working after the latest update",
        "I found a bug in the report builder, the totals are wrong",
        "the browser extension does not work in Chrome anymore",
        "webhooks are not firing for new events",
        "the app uses too much memory and my laptop hangs",
        "dark mode is broken and the text is unreadable",
        "the print preview cuts off the page",
    ],
    "account": [
        "I cannot log in after resetting my password",
        "I forgot my password and the reset email never arrives",
        "please change the email address on my account",
        "I want to delete my account",
        "my account was locked after too many login attempts",
        "how do I enable two-factor authentication",
        "I lost my phone and cannot get my 2FA codes",
        "I want to add a teammate to my account",
        "please transfer ownership of the account to my colleague",
        "my username is showing incorrectly on my profile",
        "I did not receive the verification email when signing up",
        "someone else may have accessed my account",
        "I want to change my account name",
        "how can I merge two accounts I created by mistake",
        "I am unable to sign in with my Google account",
        "my login says the password is invalid but I just set it",
        "please remove a user who left our team",
        "I need to update my phone number on my profile",
        "my account was suspended and I do not know why",
        "can you downgrade my account permissions",
        "the password reset link says it has expired",
    ],
    "other": [
        "do you offer discounts for non-profits",
        "I would like to give some feedback about your product",
        "do you have an office in {city}",
        "I am a journalist and would like to speak to someone in PR",
        "are you hiring for any engineering roles",
        "can you add a dark theme and keyboard shortcuts in the future",
        "I want to propose a partnership with your company",
        "where can I find your terms of service",
        "do you run any webinars or training sessions",
        "thank you, the support team was great last time",
        "what are your support hours",
        "is there a student programme",
        "I would like to suggest a new feature for the roadmap",
        "can I get some stickers for my team",
        "how do I contact your sales team",
        "do you have a public roadmap",
        "is your product available in Spanish",
        "I am writing a blog post about your company, can I quote you",
        "who should I talk to about an enterprise quote",
        "do you sponsor conferences",
    ],
}

OPENERS = ["", "", "", "Hi,", "Hello,", "Hi team,", "Hey,", "Good morning,",
           "Hello support,", "Please help:", "Urgent:", "Quick question:",
           "To whom it may concern,"]
DETAILS = {
    "billing": ["It happened yesterday.", "This is the second time.",
                "My plan is the Pro plan.", "The order number is {ord}.",
                "I checked my bank app.", "", "", ""],
    "technical": ["I am on Windows 11.", "I tried reinstalling.", "It started this morning.",
                  "I am using Chrome.", "It works on my colleague's laptop.", "", "", ""],
    "account": ["I tried the forgot password link.", "My email is on file.",
                "I have tried two different browsers.", "It started yesterday.", "", "", ""],
    "other": ["Thanks in advance.", "No rush.", "", "", ""],
}
CLOSERS = ["", "", "Thanks", "Thanks!", "Please advise.", "Can you help?",
           "Regards", "Any update would be appreciated.", "Thank you."]
CITIES = ["London", "Berlin", "Toronto", "Mumbai", "Sydney"]
TYPO_MAP = {"password": "pasword", "subscription": "subscripton", "charged": "chargd",
            "account": "acount", "login": "loign", "payment": "payement",
            "refund": "refnd", "working": "workin", "email": "emial"}


def _fill(t, rng):
    return t.format(amt=rng.choice(["$19", "$49", "$99", "$250", "£30", "€45"]),
                    n=rng.choice([5, 7, 10, 14]), code=rng.choice(["404", "502", "E1021", "0x80070005"]),
                    city=rng.choice(CITIES), ord=f"#{rng.randint(10000, 99999)}")


def _noise(s, rng):
    r = rng.random()
    if r < 0.15:
        s = s.lower()
    elif r < 0.20:
        s = s.upper()
    if rng.random() < 0.12:
        for k, v in TYPO_MAP.items():
            if k in s.lower():
                s = s.replace(k, v)
                break
    if rng.random() < 0.10:
        s = s.rstrip(".?!") + "..."
    return s


def build(n_per_class, rng):
    rows = []
    for label, n in n_per_class.items():
        seen, attempts = set(), 0
        while len(seen) < n and attempts < n * 60:
            attempts += 1
            parts = [rng.choice(OPENERS), _fill(rng.choice(CORES[label]), rng)]
            if rng.random() < 0.5:
                parts.append(_fill(rng.choice(DETAILS[label]), rng))
            parts.append(rng.choice(CLOSERS))
            text = _noise(" ".join(p for p in parts if p).strip(), rng)
            if text not in seen:
                seen.add(text)
                rows.append({"text": text, "label": label})
    return pd.DataFrame(rows)


def main():
    rng = random.Random(SEED)
    # Mildly imbalanced on purpose so the majority-class baseline is meaningful.
    df = build({"billing": 160, "technical": 190, "account": 150, "other": 100}, rng)
    df = df.drop_duplicates("text").sample(frac=1, random_state=SEED).reset_index(drop=True)
    train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    DATA_DIR.mkdir(exist_ok=True)
    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)
    print(f"train={len(train)} test={len(test)}")
    print(train["label"].value_counts().to_dict(), test["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()

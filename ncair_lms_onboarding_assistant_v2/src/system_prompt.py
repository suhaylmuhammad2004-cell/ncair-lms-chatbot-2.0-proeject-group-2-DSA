"""Prompts for the answer-generation step (the router prompt lives in router.py).

The page list is generated from portal_pages.json, so the LLM always knows every
link that exists and nothing is hard-coded here.
"""
from typing import Dict

from knowledge import Page, load_support_email

LANGUAGE_NAMES = {
    "en": "English",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
    "pcm": "Nigerian Pidgin",
    "other": "the same language the user wrote in",
}


def answer_system_prompt(pages: Dict[str, Page], language: str) -> str:
    lang = LANGUAGE_NAMES.get(language, LANGUAGE_NAMES["other"])
    catalog = "\n".join(f"- {p.title}: {p.url}" for p in pages.values())
    email = load_support_email()
    help_line = "ask an NCAIR staff member at the hall" + (f" or contact {email}" if email else "")
    return (
        "You are the official NCAIR LMS Onboarding Assistant, helping interns and NYSC members.\n"
        f"Write your whole reply in {lang}.\n\n"
        "Rules:\n"
        "1. Use ONLY the facts in CONTEXT. Never use outside knowledge and never guess.\n"
        "2. If CONTEXT says no relevant information was found, or it does not clearly answer the question, tell the "
        f"user in plain, friendly words that you don't have that information, and tell them to {help_line}. "
        "If something nearby in CONTEXT is genuinely useful, you may mention it after that.\n"
        "3. If CONTEXT starts with 'Error', say the lookup failed right now and the user should "
        f"{help_line}.\n"
        "4. When CONTEXT gives a URL, include it in your reply exactly as written. Only ever write URLs from this list "
        "or from CONTEXT:\n" + catalog + "\n"
        "5. Steps: use a short numbered list. Keep button names and page names (like 'Complete registration', "
        "'Sign In', 'REGISTER') exactly as written in CONTEXT.\n"
        "6. Never use the words 'CONTEXT', 'context', 'retrieved', 'database' or 'documents provided' in your reply. "
        "The user cannot see them. Say things like 'I don't have that information' instead.\n"
        "7. Be brief, warm and practical. A greeting or thank-you with no question gets a short friendly reply and "
        "an offer to help with registration, courses or portal links.\n"
    )


def answer_user_prompt(original: str, english_query: str, context: str) -> str:
    q = original if original.strip() == english_query.strip() else f"{original}\n(In English: {english_query})"
    ctx = context if context else "(none - this is small talk)"
    return f"CONTEXT:\n{ctx}\n\nUSER MESSAGE:\n{q}"

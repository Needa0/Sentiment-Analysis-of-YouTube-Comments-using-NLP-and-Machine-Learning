"""Generate a multilingual sample dataset of YouTube comments.

This produces a structured CSV using the same schema the YouTube Data API
collector writes, so the cleaning / preprocessing / language-classification
pipeline is verifiable end-to-end without requiring a live YouTube Data API key.

Comments span the languages handled by the project — English, Hindi
(Devanagari), Bengali and code-mixed (mixed scripts) — and include some emojis /
punctuation so the cleaning and normalisation stage has realistic input. Run
with::

    python data/generate_sample_dataset.py
"""

from __future__ import annotations

import csv
import itertools
import os

# Topics referenced in the comments (kept generic / content-neutral).
TOPICS_EN = ["video", "song", "tutorial", "movie scene", "recipe", "match"]
TOPICS_HI = ["वीडियो", "गाना", "ट्यूटोरियल", "फिल्म", "रेसिपी", "मैच"]
TOPICS_BN = ["ভিডিও", "গান", "টিউটোরিয়াল", "সিনেমা", "রেসিপি", "ম্যাচ"]

POS_EMOJIS = ["😍", "🔥", "👍", "❤️", "😀", "🎉"]
NEG_EMOJIS = ["😢", "😡", "👎", "💔", "😞", "😠"]
NEU_EMOJIS = ["", "🙂", "😯", ""]

# Sentiment templates per language. {t} is replaced by a topic.
TEMPLATES = {
    "English": {
        "positive": [
            "This {t} is absolutely amazing, I loved it so much {e}",
            "Such a fantastic {t}, please make more like this {e}",
            "Best {t} I have ever seen, truly wonderful work {e}",
            "Really enjoyed this {t}, great job and keep it up {e}",
        ],
        "negative": [
            "This {t} was terrible and a complete waste of time {e}",
            "I really hated this {t}, very disappointing and boring {e}",
            "Worst {t} ever, the quality is awful {e}",
            "This {t} made me angry, totally useless content {e}",
        ],
        "neutral": [
            "This is a {t} about the topic {e}",
            "Watched the {t} today, it was okay overall {e}",
            "The {t} is average, nothing special to mention {e}",
            "Here is another {t} on the channel {e}",
        ],
    },
    "Hindi": {
        "positive": [
            "यह {t} बहुत शानदार है मुझे बहुत पसंद आया {e}",
            "बहुत बढ़िया {t} ऐसे और बनाओ {e}",
            "सबसे अच्छा {t} जो मैंने देखा है शानदार काम {e}",
        ],
        "negative": [
            "यह {t} बहुत खराब था समय की बर्बादी {e}",
            "मुझे यह {t} बिल्कुल पसंद नहीं आया बहुत बुरा {e}",
            "सबसे खराब {t} गुणवत्ता बहुत खराब है {e}",
        ],
        "neutral": [
            "यह एक {t} है {e}",
            "आज {t} देखा ठीक ठाक था {e}",
            "{t} औसत है कुछ खास नहीं {e}",
        ],
    },
    "Bengali": {
        "positive": [
            "এই {t} টা দারুণ আমি খুব ভালোবেসেছি {e}",
            "খুব সুন্দর {t} আরো বানাও {e}",
            "সেরা {t} যা দেখেছি অসাধারণ কাজ {e}",
        ],
        "negative": [
            "এই {t} টা খুব বাজে সময় নষ্ট {e}",
            "আমি এই {t} টা একদম পছন্দ করিনি খুব খারাপ {e}",
            "সবচেয়ে বাজে {t} মান খুব খারাপ {e}",
        ],
        "neutral": [
            "এটা একটা {t} {e}",
            "আজ {t} দেখলাম মোটামুটি ছিল {e}",
            "{t} টা সাধারণ বিশেষ কিছু নয় {e}",
        ],
    },
    # Code-mixed comments deliberately mix Latin script with Devanagari/Bengali
    # tokens so they are classified as Code-Mixed (multi-script) text.
    "Code-Mixed": {
        "positive": [
            "This {t} bahut accha tha मस्त I loved it {e}",
            "Yaar yeh {t} is superb एकदम zabardast {e}",
            "Ei {t} ta darun দারুণ really enjoyed it {e}",
        ],
        "negative": [
            "This {t} bilkul bekaar tha बहुत waste of time {e}",
            "Yeh {t} was so bad मुझे pasand nahi aaya {e}",
            "Ei {t} ta khub baje খারাপ totally disappointing {e}",
        ],
        "neutral": [
            "Yeh ek {t} hai ठीक about the topic {e}",
            "Aaj {t} dekha মোটামুটি it was okay {e}",
            "Ei {t} ta average সাধারণ nothing special {e}",
        ],
    },
}

SENTIMENT_EMOJI = {
    "positive": POS_EMOJIS,
    "negative": NEG_EMOJIS,
    "neutral": NEU_EMOJIS,
}

TOPICS_BY_LANG = {
    "English": TOPICS_EN,
    "Hindi": TOPICS_HI,
    "Bengali": TOPICS_BN,
    "Code-Mixed": TOPICS_EN,  # code-mixed reuses latin topic words
}


def build_rows():
    rows = []
    cid = itertools.count(1)
    for lang, sentiments in TEMPLATES.items():
        topics = TOPICS_BY_LANG[lang]
        for sentiment, templates in sentiments.items():
            emojis = SENTIMENT_EMOJI[sentiment]
            for ti, topic in enumerate(topics):
                for tj, template in enumerate(templates):
                    emoji_char = emojis[(ti + tj) % len(emojis)]
                    comment = template.format(t=topic, e=emoji_char).strip()
                    rows.append(
                        {
                            "comment_id": f"c{next(cid):04d}",
                            "video_id": f"VID_{lang[:2].upper()}",
                            "author": f"user_{lang[:2].lower()}_{ti}{tj}",
                            "comment": comment,
                            "like_count": (ti * 3 + tj) % 17,
                        }
                    )
    return rows


def main():
    out_path = os.path.join(os.path.dirname(__file__), "sample_comments.csv")
    rows = build_rows()
    fieldnames = ["comment_id", "video_id", "author", "comment", "like_count"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()

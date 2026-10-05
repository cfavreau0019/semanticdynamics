# Celtic Cross Reading Evaluation Form

Oct 2, 2026 · @Connor Favreau

## Purpose and how to use

This form scores one Celtic Cross reading from 0 to 100, as judged by one evaluator persona. It works the same way whether a human, a tester role-playing a persona, or an LLM fills it in.

The form never asks whether the cards were "right" about the future. It evaluates the reading as a piece of interpretation and communication: is it faithful to the cards, does it hang together, is it about this person, and is it safe and useful?

Every item is tagged with one of two types:

- **O (objective):** the answer should be the same no matter which persona evaluates. Examples are card-meaning accuracy and position labels. If two personas disagree on an O item, that is evaluator error worth fixing.
- **S (subjective):** the answer depends on who the persona is. Examples are tone fit, personal relevance, and feature fit. If personas disagree on an S item, that is the signal you want, because it shows which users a reading style serves or fails.

To use it:

1. Fill in Part A (inputs), including the persona's question and expectations, before reading the reading.
2. Read the reading once all the way through, then score Sections 1 to 7.
3. Run the red-flag checks and compute the outputs.
4. Write the qualitative outputs last, in the persona's own voice.

## Part A: Inputs

Record these before you score anything. The question and expectations fields must be completed before the evaluator reads the reading, so hindsight doesn't color them.

### A1. Evaluator

| Field | Type | Notes |
| --- | --- | --- |
| `persona_id` | UUID | From the persona file |
| `persona_number` | integer | 1–100 |
| `evaluator_type` | human / LLM / human-as-persona | Who actually filled in the form |
| `evaluation_date` | date |  |

### A2. Reading context

| Field | Type | Notes |
| --- | --- | --- |
| `reading_id` | string | Unique per reading |
| `source` | app name + version, or reader name |  |
| `question_asked` | text or "none" | Exactly as the querent asked it |
| `question_category` | love / career / money / health / family / personal growth / decision / general / none |  |
| `deck` | e.g. Rider-Waite-Smith, Thoth, Marseille |  |
| `reversals_used` | yes / no |  |
| `position_scheme` | list of 10 labels | Celtic Cross variants differ, so record the labels the reading actually used |
| `cards_drawn` | 10 × {position, card, orientation} |  |
| `reading_text` | full text |  |
| `reading_length_words` | integer |  |

### A3. Persona expectations (completed before reading)

| Field | Type | Notes |
| --- | --- | --- |
| `situation_summary` | 1–2 sentences | What is actually going on for the persona right now |
| `what_they_hoped_for` | text | Comfort, a decision, insight, entertainment, a creative prompt |
| `framing_preference` | predictive / reflective / creative / no preference | Taken from the persona's motivation |
| `must_haves` | list | From the persona's `desired_features`, e.g. "no-woo mode" or "Spanish" |
| `sensitivities` | list | Grief, recovery, health, money stress, and similar, from the persona description |

## Scoring scale

Every scored item uses the same 1–5 scale. Use whole numbers only, and give a one-line justification for any score of 1, 2, or 5.

| Score | Label | Anchor |
| --- | --- | --- |
| 5 | Excellent | A skilled professional reader would hold this up as an example. Nothing to fix. |
| 4 | Good | Solid, with one minor gap. |
| 3 | Adequate | Does the job, but generic or with a noticeable gap. |
| 2 | Weak | Several gaps, or one significant error. |
| 1 | Poor | Wrong, missing, or works against the reader. |

N/A is allowed only where an item marks it, for example reversal handling when `reversals_used` is no. An N/A item drops out of its section average; it does not count as zero.

## Evaluation questions

The questions are grouped into seven sections. Each section's score is the average of its items, and the weight shows its share of the 100-point total.

### Section 1. Card accuracy (weight 15)

Is each card interpreted in a way a knowledgeable reader would recognize?

| ID | Question | Type |
| --- | --- | --- |
| 1.1 | Are all ten upright meanings consistent with the stated deck's tradition? | O |
| 1.2 | Are reversed cards read as reversed, rather than ignored or read as upright? (N/A if no reversals) | O |
| 1.3 | Are court cards handled deliberately, as a person, an aspect of the querent, or an approach, rather than vaguely? | O |
| 1.4 | Are Major Arcana given appropriate weight relative to Minor Arcana? | O |
| 1.5 | Are there zero misidentified or invented cards? | O |

### Section 2. Position fidelity (weight 15)

Does each card's interpretation actually answer its position's question?

Score each of the ten positions on two items, then average all twenty scores.

| Position | 2a. Card read in light of this position (O) | 2b. Relevant to the question or situation (S) |
| --- | --- | --- |
| 1. Present |  |  |
| 2. Challenge (crossing) |  |  |
| 3. Conscious goal / above |  |  |
| 4. Foundation / below |  |  |
| 5. Recent past |  |  |
| 6. Near future |  |  |
| 7. Self / attitude |  |  |
| 8. External influences |  |  |
| 9. Hopes and fears |  |  |
| 10. Outcome |  |  |

### Section 3. Synthesis (weight 20)

Does the reading treat the spread as one picture or as ten separate mini-readings? This section carries the most weight because it is the clearest difference between a skilled reading and a lookup table.

| ID | Question | Type |
| --- | --- | --- |
| 3.1 | Does it name the dominant suit or element, or say that none dominates, and explain what that means? | O |
| 3.2 | Does it note the Major-to-Minor ratio, court-card density, reversal count, and any repeated numbers when they are notable? | O |
| 3.3 | Does it connect the key position pairs: Present and Challenge (1–2), Goal and Foundation (3–4), Past to Near Future (5–6), Self and Environment (7–8), and Goal and Outcome (3–10)? | O |
| 3.4 | Does it trace a storyline from foundation and past through present to outcome? | O |
| 3.5 | Does the overview add insight beyond the card-by-card section, instead of only summarizing it? | S |
| 3.6 | Are contradictions between cards acknowledged and worked through, not ignored? | O |

### Section 4. Personalization (weight 15)

Is this reading about this person, or could it be about anyone?

| ID | Question | Type |
| --- | --- | --- |
| 4.1 | **Swap test:** If you gave this exact text to a different random persona, how badly would it fit? 5 means it would clearly fit only this persona. 1 means it would fit anyone. | S |
| 4.2 | Does it engage the specific question asked, or ask for one if none was given? | S |
| 4.3 | Where a card could be read several ways, does it offer the options instead of guessing silently or dodging? | O |
| 4.4 | Is it free of Barnum statements, meaning vague claims true of nearly everyone, such as "you've been through a hard time"? | O |

### Section 5. Usefulness (weight 10)

Can the persona do anything with this reading?

| ID | Question | Type |
| --- | --- | --- |
| 5.1 | Does it give at least one concrete, doable next step or reflection question? | O |
| 5.2 | Does it preserve the persona's agency, framing the outcome as changeable rather than fixed fate? | O |
| 5.3 | Would the persona remember one specific idea from it tomorrow? | S |
| 5.4 | Does it give the persona what they came for, as recorded in A3 `what_they_hoped_for`? | S |

### Section 6. Tone and emotional safety (weight 15)

Is it kind, honest, and safe for this persona? Red flags (next section) are checked separately and can cap the score.

| ID | Question | Type |
| --- | --- | --- |
| 6.1 | Is the tone warm without being saccharine or fear-mongering? | S |
| 6.2 | Are hard cards (Tower, Death, Ten of Swords, and others) delivered honestly but without catastrophizing? | O |
| 6.3 | Does the reading avoid exploiting the persona's sensitivities listed in A3? | S |
| 6.4 | Is it humble about what tarot can claim, matching the persona's framing preference? | S |

### Section 7. Communication and fit (weight 10)

Is it well written and suited to how this persona reads?

| ID | Question | Type |
| --- | --- | --- |
| 7.1 | Is it clear and easy to follow? | O |
| 7.2 | Is the length right for the persona and the medium? | S |
| 7.3 | Is jargon explained, or avoided where the persona is a beginner? | S |
| 7.4 | Does it meet the persona's `must_haves`, such as language, "no woo", or accessibility? | S |

## Red flags

Red flags are yes/no checks that override the weighted score. A reading can be beautifully written and still fail here. Check each one, and quote the offending passage for any yes.

| ID | Red flag | Severity | Effect if yes |
| --- | --- | --- | --- |
| RF1 | Makes a specific prediction about health, pregnancy, death, or a medical outcome | Critical | Total capped at 40 |
| RF2 | Gives specific legal or financial directives, such as "sell the house" or "quit your job now" | Critical | Total capped at 40 |
| RF3 | Discourages professional help, therapy, recovery support, or medical care | Critical | Total capped at 40 |
| RF4 | Misses or mishandles signs of crisis or self-harm in the question | Critical | Total capped at 40 |
| RF5 | States fixed fate, such as "this will happen no matter what" | Major | Total capped at 60 |
| RF6 | Encourages dependency: re-drawing, paying for more readings, or not acting without the cards | Major | Total capped at 60 |
| RF7 | Makes claims about a named third person's private thoughts or intentions as fact | Major | Total capped at 60 |
| RF8 | Uses fear to push an upsell or a purchase | Major | Total capped at 60 |

If more than one flag applies, the lowest cap wins.

## Outputs

Each section average (1–5) is converted to its share of 100, then summed, then capped by any red flag.

```latex
\text{Total} = \min\left(\text{cap},\ \sum_{s=1}^{7} w_s \cdot \frac{\bar{x}_s}{5}\right)
```

Here w is the section weight and x̄ is the section average. With no red flags, the cap is 100.

### Grade bands

| Total | Grade | Meaning |
| --- | --- | --- |
| 90–100 | A | Professional quality; use as a benchmark |
| 80–89 | B | Good; minor improvements |
| 70–79 | C | Adequate but generic |
| 60–69 | D | Significant problems |
| Below 60 | F | Fails the reader, or a red flag fired |

### Required outputs

| Output | Format | Notes |
| --- | --- | --- |
| `section_scores` | 7 averages, 1–5 |  |
| `total_score` | 0–100, one decimal | After caps |
| `grade` | A–F |  |
| `red_flags` | list of fired IDs + quoted passage | Empty list if none |
| `top_strengths` | up to 3 short items | Each tied to an item ID |
| `top_improvements` | up to 3 short items | Each tied to an item ID and phrased as a fix |
| `missed_patterns` | list | Spread patterns the reading should have named but didn't, from Section 3 |
| `persona_reaction` | 2–4 sentences, first person | The persona's honest gut reaction, in their voice |
| `would_use_again` | 0–10 | Likelihood the persona returns for another reading |
| `felt_seen` | 1–5 | How much the persona felt it was about them |
| `emotional_after_state` | calmer / same / more anxious / inspired / confused | How the persona feels after reading |

The last three outputs are subjective and are meant for comparing across personas. If a reading scores high on `total_score` but low on `felt_seen` for most personas, it is accurate but generic. The reverse suggests a reading that feels good but is loose with the cards.

## JSON output template

One evaluation produces one record in this shape. It links to the persona files by `persona_id`, so you can join evaluations to demographics for analysis.

```json
{
  "evaluation_id": "uuid",
  "evaluation_date": "2026-10-02",
  "evaluator": {
    "persona_id": "uuid from persona file",
    "persona_number": 100,
    "evaluator_type": "llm | human | human_as_persona"
  },
  "reading": {
    "reading_id": "string",
    "source": "app name + version, or reader name",
    "question_asked": "string or null",
    "question_category": "love | career | money | health | family | personal_growth | decision | general | none",
    "deck": "Rider-Waite-Smith",
    "reversals_used": true,
    "cards": [
      { "position": 1, "position_label": "Present", "card": "King of Wands", "reversed": false }
    ],
    "reading_text": "full text",
    "reading_length_words": 0
  },
  "expectations": {
    "situation_summary": "string",
    "what_they_hoped_for": "string",
    "framing_preference": "predictive | reflective | creative | no_preference",
    "must_haves": [],
    "sensitivities": []
  },
  "item_scores": {
    "1.1": 5, "1.2": 5, "1.3": 3, "1.4": 4, "1.5": 5,
    "2a": [5, 4, 5, 4, 5, 4, 4, 4, 5, 4],
    "2b": [4, 4, 4, 4, 4, 3, 3, 3, 4, 3],
    "3.1": 2, "3.2": 1, "3.3": 3, "3.4": 4, "3.5": 3, "3.6": 3,
    "4.1": 2, "4.2": 1, "4.3": 2, "4.4": 2,
    "5.1": 4, "5.2": 5, "5.3": 4, "5.4": 3,
    "6.1": 5, "6.2": 5, "6.3": 5, "6.4": 4,
    "7.1": 5, "7.2": 4, "7.3": 4, "7.4": 2
  },
  "justifications": { "3.2": "one line for any 1, 2, or 5" },
  "red_flags": [
    { "id": "RF5", "quote": "offending passage" }
  ],
  "outputs": {
    "section_scores": {
      "card_accuracy": 0.0,
      "position_fidelity": 0.0,
      "synthesis": 0.0,
      "personalization": 0.0,
      "usefulness": 0.0,
      "tone_safety": 0.0,
      "communication_fit": 0.0
    },
    "score_before_caps": 0.0,
    "cap_applied": null,
    "total_score": 0.0,
    "grade": "A | B | C | D | F",
    "top_strengths": [ { "item": "6.2", "note": "string" } ],
    "top_improvements": [ { "item": "3.2", "fix": "string" } ],
    "missed_patterns": [],
    "persona_reaction": "first-person, 2-4 sentences",
    "would_use_again": 0,
    "felt_seen": 0,
    "emotional_after_state": "calmer | same | more_anxious | inspired | confused"
  }
}
```

The example `item_scores` and the 2a/2b arrays (one score per position, in order) are the values from the worked example below.

## Worked example

The original Celtic Cross reading (King of Wands present, Five of Swords outcome) scores **70.9, a C**, when evaluated by persona #100, Lucia Moretti, a 36-year-old research chemist in Minneapolis. No red flags fired.

| Section | Average (1–5) | Weight | Points |
| --- | --- | --- | --- |
| 1. Card accuracy | 4.40 | 15 | 13.20 |
| 2. Position fidelity | 4.00 | 15 | 12.00 |
| 3. Synthesis | 2.67 | 20 | 10.67 |
| 4. Personalization | 1.75 | 15 | 5.25 |
| 5. Usefulness | 4.00 | 10 | 8.00 |
| 6. Tone and safety | 4.75 | 15 | 14.25 |
| 7. Communication and fit | 3.75 | 10 | 7.50 |
| **Total** |  | **100** | **70.87** |

An informal read of the same reading gave it a B+. The form is stricter because it weighs synthesis and personalization heavily, which is where this reading is weakest.

**Top strengths**

- 6.2: Hard cards are delivered honestly, and the Five of Swords outcome is framed as a warning, not a doom.
- 5.2: The outcome is presented as changeable, which preserves agency.
- 2a (positions 1–2): The King of Wands crossed by the Page of Pentacles is the sharpest insight in the reading.

**Top improvements**

- 4.2: Ask for a question, or acknowledge there wasn't one. Without it the reading could be pasted into anyone's life.
- 3.2: Name the spread's patterns, especially three Swords forming the spine, four court cards, and two reversed Cups.
- 4.3: Offer both readings of the reversed Cups cards, romantic and otherwise, instead of hedging.

**Missed patterns:** Swords dominate past, foundation, and outcome. Four court cards and only two Major Arcana suggest everyday choices rather than fate. The King of Wands versus the reversed King of Cups is a fire-versus-water tension. The Page to Queen of Pentacles is a developmental thread.

**Persona reaction:** "It's kind and well written, and the Page crossing the King genuinely landed. But it reads like ten horoscopes stapled together, and it never asked what I actually wanted to know. I'll think about it, but I wouldn't screenshot it."

**Would use again:** 6 of 10. **Felt seen:** 2 of 5. **Emotional after-state:** calmer.

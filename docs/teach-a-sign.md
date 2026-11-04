# Teach a sign: how it works and how well

SignBridge's trained models only know the words in their datasets (262 ISL, 250 ASL). Words
like *help*, *pain* or *fever* are missing from every public ISL dataset we found. Teaching
fills that gap: someone records a sign 5 times, and SignBridge stores it on the clinic server
for everyone using that sign language.

**Method.** The recognition network turns each recording into a 320-number fingerprint. A new
signing is compared with every taught example (cosine similarity). A taught word competes with
the known words on one scale: `score = 20 x similarity - 19`, against the model's
log-probabilities. The weights were chosen on a held-out test so known words stay accurate.

**Measured (ISL model retrained with 30 signs hidden, then taught those 30):**

| Examples taught | Correct on never-trained signs |
|---|---|
| 1 | 80.3% |
| 3 | 92.7% |
| 5 | 95.2% |

Mixed with all known words, using the chosen weights: known words 95.5% (was 97.6%), taught
words 71% as the first guess. When the first guess is wrong, the right word is usually offered
as a one-tap choice.

**Limits.** On noisy phone recordings (Google's ASL data), teaching worked for only about
25–50% of attempts when a different person signed than the one who taught it. It works best
when recordings are clear, and when the same person teaches and uses the sign. More examples
from more people help.

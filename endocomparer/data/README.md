# Data: the JSONL schema

EndoComparer trains and runs on **items**: two videos, one question stem, four options. Items come either from the
EndoDiffVQA dataset (`python -m endocomparer.data.from_dataset`) or from a JSONL file you write, one JSON
object per line. `example_train.jsonl` holds seven **synthetic** rows (invented ids, questions and video paths) that
cover presence, state, count, attribute and skill questions; it exercises the target renderer, and
cannot be trained on because its videos do not exist.

## Fields

| field | type | needed for | meaning |
|---|---|---|---|
| `id` | str | all | unique item id |
| `data_source` | str | all | subset name; `chole_score` / `multibypass_skill` (or `category: Skill`) select the skill frame budget (64 frames) instead of 16 |
| `template_id` | str | all | question template: `A1 A5 A6 T1 T6` (presence), `A2 A3 A4 T3 C1 C2 C4` (attribute values), `T5` (count), `S1 S2` (skill). Decides how facts render and which Strict group the item is scored in. Unknown ids need a `template_kind` field (`presence`, `attribute`, `count` or `skill`). |
| `category` | str | optional | `Anatomy` / `Tool` / `Action` / `Skill` |
| `attribute` | str | optional | free text (`presence`, `color`, `count`, …) |
| `question` | str | all | the question **stem** only, without the options |
| `options` | {a,b,c,d: str} | all | option texts, in the order they are shown for the `full` control |
| `answer_letter` | a–d | training, scoring | gold letter |
| `option_codes` | {a,b,c,d: str} | optional | `correct`, `D1` (video-swap distractor), `D2` (same/different collapse), `D3` (magnitude), `D4` (substitute); used for the swap answer and the "swap error" statistic |
| `swap_answer_letter` | a–d or null | training (swap augmentation), swap control | gold letter once the two videos are exchanged: the same letter for a symmetric answer (both / neither / same), the `D1` option otherwise, `null` if no option states it. Derived from `option_codes` / the option texts when absent. |
| `cell` | str | training | answer type of the gold: `BOTH ONLY_A ONLY_B NEITHER` (presence), `SAME DIFFERENT ONLY_A ONLY_B` (attribute), `SAME MORE_A MORE_B NEITHER` (count), `SAME CLEAR_A CLEAR_B SLIGHT_A SLIGHT_B` (skill) |
| `value_a`, `value_b` | str or null | training (attribute, count) | the attribute value / instrument count seen in Video 1 / Video 2; `null` = the subject is not visible in that video |
| `score_a`, `score_b` | int or null | training (skill) | the expert score of Video 1 / Video 2 |
| `subject_keys` | list[str] | training (`A6`) | the question's subject; for `A6` the last entry is the state word ("inflamed") used in the statements |
| `video1`, `video2` | str | all | clip paths, absolute or relative to `--video-root` (default: the JSONL's directory) |
| `frames1`, `frames2` | int | optional | native frame counts; probed with decord when absent |

**Per-video statements** (PtC-CoT targets) are rendered from these fields by `endocomparer.ptc.per_video_halves`:

| template kind | Video k statement |
|---|---|
| presence `A1` / `T1` / `T6` | `visible` / `not visible`, `used` / `not used`, `engaged` / `not engaged`, from `cell` |
| presence `A6` | `subject_keys[-1]` or `not {subject_keys[-1]}`, from `cell` |
| count `T5` | `{value_k} instrument(s)` |
| attribute | `value_k`, or `not visible` when `value_k` is null |
| skill | `score {score_k}` |

Inference needs only `id`, `data_source`, `template_id`, `question`, `options`, `video1`, `video2`
(and `answer_letter` for scoring). Option texts drive the Same / Different classes of the metrics through the
template wording (`Yes —` prefix for symmetric answers, `Video 1 only` / `Video 2 only`, `more instruments in Video k`,
`clearly / slightly better in Video k`), so keep the dataset phrasing for items you want scored hierarchically.

## From the dataset

The dataset is not public yet (coming soon). With a dataset directory (layout in `endodiffvqa/README.md`):

```bash
python -m endocomparer.data.from_dataset --dataset-root data/EndoDiffVQA --splits train,val --out data/trainval.jsonl
python -m endocomparer.data.from_dataset --dataset-root data/EndoDiffVQA --splits test --out data/test.jsonl
```

Rows are written in dataset order; `endocomparer.train` shuffles them with its seed, so the paper run's instances
are reproduced exactly from the converted file (checked against the original run's training configuration).

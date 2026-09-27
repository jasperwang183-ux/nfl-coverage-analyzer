---
name: coverage-knowledge
description: Football coverage knowledge for reading NFL defenses - what each coverage family shows pre-snap, what it does post-snap, how it disguises as another, and the position/movement tells that separate look-alikes, mapped to the Big Data Bowl labels. Use whenever touching shell/MOF rules, deep thresholds, disguise components, confidence rationale, card reasons, the coverage classes, or when judging whether a film read (shown, played, exact call, disguise) is right.
---

# Coverage Knowledge

Paraphrased from coaching articles (sources at the end) plus what the project's own film
checks have shown. Everything here is about positions and movement, because that is all
the pipeline sees: canonical frames at 10 Hz, offense moving left to right, depth = x past
the LOS, lateral = y off the formation center (`src/nfl_coverage/data/standardize.py`).

**Depths are mostly unsourced.** The articles give almost no yardages. Numbers marked
*(src)* are stated by a source; *(prior)* are common coaching values used as priors;
the rest are the repo's own thresholds. Never tune a threshold to a *(prior)* without a
verify run.

## Labels and what they cover

| BDB label | Families it absorbs | Deep players post-snap | Man/zone |
|---|---|---|---|
| COVER_0_MAN | Cover 0, zero pressure, "zero with two rats" (still 0: rats are short) | 0 | man |
| COVER_1_MAN | man-free, Rat/Hole, Robber, Spy, Bracket, Double Robber, Cover 1 pressures (Omaha) | 1, deep middle | man |
| COVER_2_MAN | 2-Man / 2-Man Under / "Cover 5", Swipe, MEG-style two-high man | 2 halves | man |
| COVER_2_ZONE | Cover 2, Tampa 2, Red 2, Green 2, Mug Green 2, rotated/bear-front Cover 2 pressures | 2 halves (Tampa: MIKE runs the middle, looks like 3) | zone |
| COVER_3_ZONE | Cover 3, Sky, Buzz, Cloud, Sting, Match/RIP-LIZ, fire zone (3-deep 3-under), HOT (3-deep 2-under), Cover 3 sims | 3 thirds | zone (match plays man-like) |
| COVER_4_ZONE | Quarters, pattern-match quarters, MOD, Thumbs, Robber, Switch, 2-Read/Palms (can play like 2), Saban Cover 7/8 | 4 quarters (2-Read: halves) | zone (MOD/MEG man-like) |
| COVER_6_ZONE | Quarter-quarter-half, split-field; quarters to trips/strength, half to the single side | 3 (can convert to 3 thirds on a #3 vertical) | zone |

Ambiguous by design, so expect label noise and do not overfit these: Tampa 2 vs Cover 3,
2-Read vs Cover 2/4, MEG vs 2-Man, Steal (two-high look, single-high play), Cover 6 on a #3
vertical, Match Cover 3 vs Cover 1. The label is the coverage, never the blitz.

## Families

### Two-high shell (Cover 2 zone, Tampa 2, 2-Man, 2-Read, Quarters, Cover 6)
All show two high safeties and an open middle, so **the shown shell cannot separate
them**; the split happens after the snap.
- **Cover 2 zone.** Safeties about 10-14 yd *(src)* just outside the end man on the line,
  settling 15-25 yd *(src)* once the play develops. Corners 3-5 yd off *(src)*, squatting in
  the flat. Five under, two deep halves. **Red 2** (red zone): safeties shuffle in place
  instead of bailing. **Green 2** (passing downs): safeties deeper, corners sink and trail.
  In "flip" formations the safety takes the flat and the corner the deep half, so role
  comes from post-snap depth, not from who the player is.
- **Tampa 2.** Cover 2 look; the MIKE, starting in the box, runs down the middle past
  about 10-12 yd *(prior)* while both safeties widen. Looks like 3-deep after the snap.
- **2-Man.** Two deep halves driving on the inside seams; underneath defenders trail
  their receivers low and on the inside hip, holding a small offset.
- **Quarters (Cover 4).** Corners own the outside quarter with outside leverage. **Each
  safety reads #2:** #2 vertical, the safety carries him; #2 short, the safety drives down
  and inside to double #1 or fill the run. Variants change who carries what (MOD, MEG,
  Thumbs, Robber, Switch). **2-Read/Palms:** on #2 short/out the corner comes down on #2 and
  the safety widens over #1, so it plays like Cover 2 on that side.
  Michigan State's version *(src, college)*: safeties 7-9 yd deep, about 2 yd inside a
  split #2 (2 yd outside an attached #2 or the edge when #2 is the back); the safety
  backpedals and reads, and only picks #2 up man-style once #2 has stemmed about 7 yd
  vertical. On #2 releasing straight outside he turns to #1 over the top. The backside
  safety on run away screws down to linebacker depth and robs crossers.
- **Cover 2 Trap / Eagle.** The corner ignores #1, parks flat-footed at shallow depth
  reading the backfield for a flat threat, and sinks to a deep quarter if none comes, so it
  can finish looking like Cover 4. The tell is the parked corner, not one backpedaling with #1.
- **Cover 6.** Quarters to the trips/strength side, a half to the single side. On a #3
  vertical the trips-side free safety runs to the middle third: Cover 3 by movement. It is
  also a check: Ohio State played Cover 2 to a nub side (the corner becomes the shallow force
  player, the safety widens to the hash) and quarters to the other. The halves behave
  differently, so read them separately.
- **Solo (trips check from quarters).** The backside safety leaves the lone receiver's
  corner and runs across to carry #3 vertical; the lone corner plays tight man with no help.
  After the snap it can look like Cover 3 or Cover 1 rotated to trips; unlike Cover 3 the
  lone corner does not bail and the post player is not centered.
- **Mug Green 2** shows single-high and rotates to two-high *before* the snap; **Swipe**
  shows single-high at the snap and is two deep within about 1 s (the corner bails to a
  half, a box safety runs out to trail the wideout).

### Single-high shell (Cover 1, Cover 3, Cover 0 with a deep look)
- **Cover 1.** One free safety in the deep middle. Corners leverage by receiver split
  (inside the numbers: outside leverage; outside: inside leverage; else head-up) and
  mirror their man. Rat/Hole puts a linebacker in the short middle; **Robber** does it with
  a safety, often after spinning from a two-high shell. A defender crossing the formation
  with motion points to man.
- **Cover 3.** One deep middle safety, corners bail to deep thirds with a cushion
  whatever the receiver does. Sky/Buzz rotate a safety down to curl-flat or hook; Cloud
  rotates the corner down and the safety over the top. Often shown as two-high and spun at
  the snap. Match/RIP-LIZ plays man-like once routes declare.
- **Cover 0.** Nobody deep. Mugged A-gaps and pressed corners are shared with Cover 1
  and sim pressure; the decider is whether anyone reaches deep middle depth after the snap.

### Disguise-first defenses
Some coordinators build on shells that carry almost no signal. Jim Knowles (college) lines
up all five DBs about 10 yd off from a three-high base and plays Cover 3 Robber, Cover 3
Cloud, two Tampa 2 versions, Cover 1, Cover 4 and zero from the same picture, varying safety
depths on purpose. Only post-snap paths say which: who drops to about 12 yd or more in the
middle, who buzzes to 5-8 yd. Against 3x1 the middle safety takes the post in Cover 3 but the
boundary safety does in man, so *which player* takes the post is itself a man/zone tell.
On such plays the shown read deserves low confidence, and a disguise flag is most justified.

### Bunch, stack and in/out rules
Banjo and in/out rules (first out / second in) make man coverage look like zone for the
first 0.5-1 s, then each defender locks onto one receiver. Michigan State's "closed trips"
answer, still called Cover 4, played bunch man-match: one "lock" corner on #2 from the snap,
the others declaring after the releases. Expect man/zone label noise against compressed sets.

### Pressure (drives the look, not the label)
Fire zone = Cover 3 behind five; HOT = Cover 3 with six; sim pressure / creeper keeps a
four-man rush while a second-level player rushes and a lineman drops; Omaha = Cover 1 with
six. **Box count and mugged players predict pressure, not the coverage.** A six-up look that
rushes four and plays three-deep shows Cover 0 and plays Cover 3. Michigan State ran one
many-high shell with six, five or four rushers behind it (the last drops both outside
linebackers); count defenders crossing the LOS in the first second, not the look. In a
3-deep 3-under zone blitz the ends drop backward while linebackers cross, three players
settle about 12 yd or deeper, and three underneath key on #2 and #3.

**Naming traps.** Call numbers are not labels: "Over Gut 6" is a Cover 3 fire zone, and a
team can call bunch man-match "Cover 4". Never map playbook text to a label by its digit.

### Fronts as priors (Tite)
4i-0-4i with apex overhangs 4-5 yd off the ball *(src)* leans zone (Cover 3/4). 4-man box +
three high: rotation likely. 5-man box + two high: Cover 2/4 family likely. An overhang
walking out onto the slot pre-snap points to man or match.

## Tells a position/movement reader can measure

1. **Rotation vs a safety doing his job.** A real two-high-to-one-high rotation moves one
   safety *to the middle*. In quarters, Cover 2 and Red 2 a safety can come down or hold
   while **the other safety stays over his own half**. Check the partner's lateral path.
2. **Read the shell after the snap, not at it.** Robber, Buzz, Sky, Cloud, Invert, Jump,
   Swipe all change the deep count in the first 1-1.5 s. The deepest defender near the
   middle at about 1-1.5 s is the shell, and it may not be the player who started deep.
3. **Deep because of a route, not a zone.** Man defenders carrying verticals end up deep:
   a Cover 1 corner on a go route is not a second high safety. A deep defender within a
   couple of yards of a receiver, moving with him, is carrying, not zoning.
4. **Man vs zone.** Man: small, sustained defender-to-receiver distance through the
   breaks (judge after about 1.5 s; In/Out, triangle and banjo swaps blur the first second).
   Zone: distance grows as receivers leave the zone; corners hold cushion and landmarks.
5. **Split field.** Judge the two halves separately; man on one side only (Steal backside,
   MEG) argues for a split-field call, not pure man or zone.
6. **Safety path follows #2.** In quarters/Cover 6 the safety's depth change tracks #2's
   (and #3's) vertical release, which a pre-planned rotation does not. Pattern match turns
   into man only after a trigger (#2 past about 7 yd; first out / first in), while true man
   mirrors from the snap: defender-to-receiver distance by frame separates the two.
7. **Rushers.** Count defenders past the LOS in the first second separately from the shell.

## What the project's film checks taught

- **Asymmetric two-high on the MOF edge.** On 1326 one safety sat 3.4 yd off the middle
  against a 3.5 yd `MOF_HALFWIDTH` (`analysis/shell.py`) with the other 8.7 yd out, so the
  reader called a post safety plus a half; both then held their halves at 13-15 yd (tell 1)
  and the analyst confirmed Cover 2 zone. The analyst also marked the shell rows (2-high
  closed, then 1-high closed) correct, so whether the "rotated" disguise card was wrong is
  unresolved.
- **The 9 yd deep line is an edge, not a bug.** A defender at 8.9 yd split 9.3 yd off the
  middle was a high safety (analyst: three-high look, 2-Man played); one at 8.8 yd over the
  middle was not (1533, analyst: 0-high). Counting side defenders at 7-9 or 8-9 yd, or
  lowering the line to 8 yd, all cut shown-MOF agreement with the labels on weeks 1-4 of both
  seasons (67.7% to 63.5-66.1%, 7,340 plays). Keep 9 yd; a defender on the line lowers the
  confidence and says so, which is the right response.
- **Man from a zone look.** 15-34-46 (analyst label): three-high zone look, 2-Man played,
  press on the single receiver, off/squat elsewhere, RB stayed in and a free underneath
  defender. On the tracks the press corner stays 1-2 yd from his man and both safeties hold
  halves at 14-16 yd; the off corner's growing cushion read like zone (tell 4 needs the
  routes to declare, and 2.5 s of tracks is short). The model said Cover 2 zone.
- **Officials at the LOS get sided.** Two static bodies about 23 yd out on each sideline
  at LOS depth (the head linesman and line judge) were tracked as defenders on 15-34-46 and
  became the "widest defenders" in the reasons. Fixed in `vision/track.py` (4 yd snap
  sideline margin, ruled-out ids stay out, sideline strays in the offense's backfield are not
  sided); with them gone the played read names the press corner.
- **Crops remove the corners**, so a safety becomes the "widest defender" and must clear
  `CORNER_DEEP_DEPTH` (12 yd, `analysis/report.py`) to count as deep.
- **Tracker duplicates** can make one deep player two (1533: two "deep" ids 2.4 yd apart
  at +2.5 s). Check pairwise distance before counting a second high safety.
- **Clip 1 (Cover 4)** is the textbook quarters bail: both safeties 12 to 15 yd and
  widening, a corner bailing 8 to 14 yd wide; `corner_bail` = no disguise is right.

## Using this in the repo

Reads and rules live in `analysis/shell.py` (shell, MOF), `analysis/report.py` (structural
read, confidence, reasons, disguise text), `analysis/disguise.py` (components) and
`vision/pipeline.py` (film). Any rule from this skill that changes a threshold, a disguise
component or a confidence level must go through the `verify` skill: the frozen holdout and
`harness.run_shell_accuracy` on tracking data, plus the analyst-confirmed film clips in
`data/processed/coverage_corrections.jsonl` (target `shell` and `exact`). A rule lands only
if it fixes misses without breaking confirmed reads. Card reasons should name the tell
("the other safety held his half, so not a rotation") so the level carries its rationale.

## Sources

Most sources are college football (the Big Ten, 2013-2022); NFL depths and habits can
differ, which is one more reason every threshold goes through verify.

breakdownsports.blogspot.com: /p/coverage.html, /p/blitz-package.html, Cover 2 fundamentals
(2016/10), Minnesota Mug Green Cover 2 (2022/03), the Cover 2 label index, Tite front
(2019/05), Cover 4 playbook (2015/05), Cover 4 front-seven leverage (2015/07), Cover 1
playbook and adjustments (2014/08), Jim Knowles primer (2022/09), MSU Cover 4 safety play
(2014/08), MSU Cover 3 package parts 1-2 (2013/10), Ohio State Solo vs Illinois (2015/11),
Oregon vs MSU closed trips (2014/09), dictating MSU's defense (2013/11), Over Gut 6 zone
blitz (2014/11), Ohio State Cover 6 (2013/12), Banjo coverage (2018/10), Don Brown's defense
(2016/03). The blog maps about 40 coverage posts; the rest (more MSU Cover 4 film, Nebraska
single-high, press technique, Wisconsin and Don Brown pressure) are unread. blitzology.com: Swipe 2-Man (2020/08), 4-2-5
quarter-quarter-half vs trips (2015/07); man vs tandem/bunch (2010/05) did not load.
matchquarters.com: Steal coverage (partly paywalled). Bleacher Report: NFL 101 2-Man.

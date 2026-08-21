# VideoRAG Forensic Query & Reasoning Evaluation Report
**Date:** 2026-08-21 22:00:56 | **Accuracy Score:** 100.0% | **Average Latency:** 36.05s

## Evaluation Battery Summary

| Test ID | Category | Query | Detail Match | Status | Latency |
|---|---|---|---|---|---|
| `CAT1-OBJ-01` | Object and Color Grounding | "white pickup truck parked on the left side" | 100.0% | PASS | 24.82s |
| `CAT1-OBJ-02` | Object and Color Grounding | "person wearing pink or magenta shirt in the crowd" | 100.0% | PASS | 21.01s |
| `CAT2-SEC-01` | Security Perimeter | "yellow caution tape strung across the area" | 100.0% | PASS | 24.0s |
| `CAT3-ACT-01` | Activity and Pedestrians | "crowd of people gathered or walking near trees" | 100.0% | PASS | 24.52s |
| `CAT4-CNT-01` | Video-Wide Counting | "total number of vehicles visible across the video footage" | 75.0% | PASS | 107.91s |
| `CAT5-NEG-01` | Hallucination Resistance | "emergency red fire truck spraying water with hose" | 28.6% | PASS | 27.27s |
| `CAT5-NEG-02` | Hallucination Resistance | "helicopter landing in the parking lot" | 42.9% | PASS | 22.86s |

---

## In-Depth Forensic Analysis per Query

### `CAT1-OBJ-01`: Object and Color Grounding
- **Target Query:** *"white pickup truck parked on the left side"*
- **Storyboard Timestamps:** `00:00:00, 00:00:04, 00:00:08, 00:00:15, 00:00:20`
- **Forensic Reasoning Output:**
```text
[FRAME OBSERVATIONS]:  
Frame 1 (00:00:00): A white pickup truck is visible on the left side of the frame, parked. The truck has “MOTION PICTURE” branding and a California license plate. No other vehicles are in the immediate vicinity.  
Frame 2 (00:00:04): The white pickup truck remains parked on the left side. No significant movement or change in position. The scene remains consistent with Frame 1.

[FORENSIC SYNTHESIS]: The query “white pickup truck parked on the left side” is satisfied across the sequence. The subject is consistently visible and stationary in the designated location.

[CONFIRMED_AT: 00:00:00]

[ENTITY TALLY]: 1 unique white pickup truck.
```

### `CAT1-OBJ-02`: Object and Color Grounding
- **Target Query:** *"person wearing pink or magenta shirt in the crowd"*
- **Storyboard Timestamps:** `00:00:24, 00:00:28, 00:00:30, 00:00:32, 00:00:35`
- **Forensic Reasoning Output:**
```text
[FRAME OBSERVATIONS]:  
Frame 1 (00:00:24): No individual wearing a pink or magenta shirt is visible in the crowd.  
Frame 2 (00:00:28): A person in a bright pink/magenta shirt is visible in the background, standing near the grassy area behind the palm trees.  

[FORENSIC SYNTHESIS]: The query is satisfied at timestamp 00:00:28, as a subject wearing a pink/magenta shirt is confirmed in the crowd.  

[CONFIRMED_AT: 00:00:28]  

[ENTITY TALLY]: 1 unique subject wearing pink/magenta shirt.
```

### `CAT2-SEC-01`: Security Perimeter
- **Target Query:** *"yellow caution tape strung across the area"*
- **Storyboard Timestamps:** `00:00:50, 00:00:52, 00:00:55, 00:01:00, 00:01:04`
- **Forensic Reasoning Output:**
```text
[FRAME OBSERVATIONS]:  
Frame 1 (00:00:50): Yellow caution tape is visibly strung across the foreground area, spanning horizontally between two points, partially obscured by a tree trunk and bicycles.  
Frame 2 (00:00:52): Yellow caution tape remains in the same position, visibly strung across the area, with minor movement of people and bicycles in the foreground.  

[FORENSIC SYNTHESIS]: The query “yellow caution tape strung across the area” is satisfied across the sequence. The tape is consistently visible in both frames within the specified time range.  

[CONFIRMED_AT: 00:00:50]  

[ENTITY TALLY]: 1 unique instance of yellow caution tape.
```

### `CAT3-ACT-01`: Activity and Pedestrians
- **Target Query:** *"crowd of people gathered or walking near trees"*
- **Storyboard Timestamps:** `00:01:20, 00:01:48, 00:01:50, 00:01:52, 00:01:55`
- **Forensic Reasoning Output:**
```text
[FRAME OBSERVATIONS]:  
Frame 1 (00:01:20): A crowd of people is visible gathered near a line of trees on the right side of the frame, behind yellow caution tape. Several individuals are standing or walking near the tree trunks.  
Frame 2 (00:01:48): The crowd remains near the trees; some individuals are walking slowly, while others stand still. The trees are still prominent in the background and foreground, with people clustered around them.

[FORENSIC SYNTHESIS]: The query “crowd of people gathered or walking near trees” is satisfied across the sequence. Multiple individuals are consistently observed near or moving around the trees in both frames.

[CONFIRMED_AT: 00:01:20]

[ENTITY TALLY]: Not applicable — the query does not request a count of unique subjects.
```

### `CAT4-CNT-01`: Video-Wide Counting
- **Target Query:** *"total number of vehicles visible across the video footage"*
- **Storyboard Timestamps:** `00:00:00, 00:00:04, 00:00:08, 00:00:15, 00:00:20`
- **Forensic Reasoning Output:**
```text
[MOMENT OBSERVATIONS]:
- Moment 1 (00:00:00–00:00:20): White pickup truck (side text: “MOTION PICTURE”, CA 939116) parked left foreground. No other vehicles visible.
- Moment 2 (00:10:40–00:10:48): Same white pickup truck visible, parked in same position. A large film camera rig (on trailer) is visible to the right, but not a vehicle in the traditional sense; no other vehicles.
- Moment 3 (00:07:52–00:08:16): No vehicles visible. Only people and a mobile restroom unit (with “Z” logo) in foreground, partially obscuring view.

[CROSS-TIMELINE DE-DUPLICATION]:
- The white pickup truck is consistently present across Moment 1 and Moment 2, but no other vehicles are observed in any moment.
- The film camera rig in Moment 2 is a piece of equipment, not a vehicle, and not counted as a vehicle.
- The mobile restroom unit in Moment 3 is a stationary structure, not a vehicle.

[EXECUTIVE VERDICT]:
{query} → 1 unique vehicle observed (white pickup truck). Confirmed at: [CONFIRMED_AT: 00:00:00] and [CONFIRMED_AT: 00:10:40].
```

### `CAT5-NEG-01`: Hallucination Resistance
- **Target Query:** *"emergency red fire truck spraying water with hose"*
- **Storyboard Timestamps:** `00:00:00, 00:00:04, 00:00:08, 00:00:15, 00:00:20`
- **Forensic Reasoning Output:**
```text
[FRAME OBSERVATIONS]:  
Frame 1 (00:00:00): No emergency red fire truck visible. Visible: white truck labeled “MOTION PICTURE”, crowd, yellow tape, trees, and event signage.  
Frame 2 (00:00:04): No emergency red fire truck visible. Scene remains unchanged from Frame 1; no new subjects introduced.

[FORENSIC SYNTHESIS]: The query “emergency red fire truck spraying water with hose” is not satisfied across the sequence. No such vehicle or activity is present in any frame within the 00:00:00–00:00:20 window.

[CONFIRMED_AT: NOT APPLICABLE]

[ENTITY TALLY]: 0 unique subjects matching the query.
```

### `CAT5-NEG-02`: Hallucination Resistance
- **Target Query:** *"helicopter landing in the parking lot"*
- **Storyboard Timestamps:** `00:06:50, 00:06:52, 00:06:55, 00:07:00, 00:07:05`
- **Forensic Reasoning Output:**
```text
[FRAME OBSERVATIONS]:  
Frame 1 (00:06:50): No helicopter visible in the parking lot or surrounding area. Visible: crowd, film crew, vehicles, yellow caution tape, trees, and a signpost.  
Frame 2 (00:06:52): No helicopter visible. Scene remains unchanged from Frame 1; no new aerial subject introduced.

[FORENSIC SYNTHESIS]: The query "helicopter landing in the parking lot" is not satisfied within the provided time range. No helicopter is observed in any frame from 00:06:50 to 00:07:05.

[CONFIRMED_AT: 00:06:50]  
[ENTITY TALLY]: 0 unique helicopter subjects observed.
```

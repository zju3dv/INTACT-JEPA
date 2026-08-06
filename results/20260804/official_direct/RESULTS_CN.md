# Window Protocol Official Direct Results

- Updated: `2026-08-04T06:51:54+08:00`
- Completion: `216/216`
- Status: `FINAL`
- Each task cell contains 3 train seeds x 3 eval seeds x 100 episodes.

## Four-task Macro SR

| Layout | Protocol | Complete tasks | Macro SR (%) | Train-seed std (pp) |
|---|---|---:|---:|---:|
| five_slot | p75 | 4/4 | 95.00 | 0.22 |
| five_slot | p77 | 4/4 | 94.75 | 0.17 |
| five_slot | p66 | 4/4 | 95.11 | 0.05 |
| four_slot | p75 | 4/4 | 94.08 | 0.75 |
| four_slot | p77 | 4/4 | 93.83 | 1.13 |
| four_slot | p66 | 4/4 | 93.28 | 1.54 |

## Task SR

| Layout | Protocol | Task | Results | Mean SR (%) | Train-seed std (pp) |
|---|---|---|---:|---:|---:|
| five_slot | p75 | pusht | 9/9 | 83.33 | 0.67 |
| five_slot | p75 | cube | 9/9 | 100.00 | 0.00 |
| five_slot | p75 | reacher | 9/9 | 97.89 | 0.19 |
| five_slot | p75 | tworoom | 9/9 | 98.78 | 0.51 |
| five_slot | p77 | pusht | 9/9 | 83.89 | 0.51 |
| five_slot | p77 | cube | 9/9 | 100.00 | 0.00 |
| five_slot | p77 | reacher | 9/9 | 96.00 | 0.33 |
| five_slot | p77 | tworoom | 9/9 | 99.11 | 0.51 |
| five_slot | p66 | pusht | 9/9 | 84.33 | 0.33 |
| five_slot | p66 | cube | 9/9 | 100.00 | 0.00 |
| five_slot | p66 | reacher | 9/9 | 97.11 | 0.38 |
| five_slot | p66 | tworoom | 9/9 | 99.00 | 0.00 |
| four_slot | p75 | pusht | 9/9 | 82.44 | 1.90 |
| four_slot | p75 | cube | 9/9 | 100.00 | 0.00 |
| four_slot | p75 | reacher | 9/9 | 97.00 | 0.00 |
| four_slot | p75 | tworoom | 9/9 | 96.89 | 1.26 |
| four_slot | p77 | pusht | 9/9 | 82.11 | 1.64 |
| four_slot | p77 | cube | 9/9 | 100.00 | 0.00 |
| four_slot | p77 | reacher | 9/9 | 95.89 | 0.69 |
| four_slot | p77 | tworoom | 9/9 | 97.33 | 2.65 |
| four_slot | p66 | pusht | 9/9 | 82.00 | 0.58 |
| four_slot | p66 | cube | 9/9 | 100.00 | 0.00 |
| four_slot | p66 | reacher | 9/9 | 96.33 | 1.20 |
| four_slot | p66 | tworoom | 9/9 | 94.78 | 6.74 |

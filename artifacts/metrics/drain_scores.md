# Drain-aware scores (drain-v1)

Window 25200-28800 s, 14942 scheduled cars (1864 zero-length), n = 3 evaluation seeds per controller (simulation noise only; one trained model per RL controller). Lag/scheduled car = mean over ALL scheduled cars of (actual completion, or 08:00 if unfinished/never inserted) minus (zero-delay completion time); lower is better.

| Controller | Lag/scheduled car (s) | SD | Due-completion | Completed (non-zero cars) | Never inserted | Teleports | TimeLoss non-zero (s) |
|---|---|---|---|---|---|---|---|
| sumo_actuated | 276.7 | 1.1 | 0.840 | 9653 | 539 | 7.7 | 337.6 |
| fixed_time | 296.0 | 6.8 | 0.827 | 9508 | 466 | 7.7 | 358.9 |
| max_pressure | 407.3 | 5.4 | 0.738 | 8480 | 1318 | 50.3 | 415.3 |
| ppo | 485.2 | 3.0 | 0.663 | 7620 | 1819 | 19.0 | 438.3 |
| ippo | 477.4 | 7.4 | 0.661 | 7597 | 1875 | 10.0 | 430.4 |
| ippo_phase_select | 501.1 | 2.8 | 0.656 | 7544 | 1658 | 712.7 | 360.2 |

## A. Labeler check against the paper's failure-label table (stored runs, multi_turn_base)

| column | run | empty (paper / mine) | under-call (paper / mine) | state mismatch (paper / mine) | failed (paper / mine) | equal |
|---|---|---|---|---|---|---|
| NFC | 4 | 59 / 59 | 19 / 19 | 68 / 68 | 146 / 146 | yes |
| IFFC | 5 | 41 / 41 | 38 / 38 | 60 / 60 | 139 / 139 | yes |
| Stop | 30 | 38 / 38 | 37 / 37 | 60 / 60 | 135 / 135 | yes |
| Ungated | 27 | 40 / 40 | 31 / 31 | 75 / 75 | 146 / 146 | yes |
| Summary | 10 | 31 / 31 | 37 / 37 | 64 / 64 | 132 / 132 | yes |
| Minimal | 42 | 17 / 17 | 41 / 41 | 71 / 71 | 129 / 129 | yes |
| Full | 78 | 20 / 20 | 35 / 35 | 68 / 68 | 123 / 123 | yes |
| Vote | 11 | 36 / 36 | 38 / 38 | 53 / 53 | 127 / 127 | yes |
| CD | 8 | 14 / 14 | 70 / 70 | 95 / 95 | 179 / 179 | yes |
| Autom. | 46 | 17 / 17 | 54 / 54 | 102 / 102 | 173 / 173 | yes |
| SFT | 15 | 2 / 2 | 59 / 59 | 109 / 109 | 170 / 170 | yes |

Labeler reproduces every column of the paper's table: True.

## B. Primary failure label per failed task, G4 runs (200 tasks each)

| label | U1 | T1 | T2 | U2 |
|---|---|---|---|---|
| empty turn | 12 | 0 | 0 | 12 |
| under-call | 46 | 41 | 40 | 46 |
| state mismatch | 54 | 90 | 91 | 53 |
| other | 0 | 0 | 0 | 0 |
| failed | 112 | 131 | 131 | 111 |
| passed | 88 | 69 | 69 | 89 |

## C. pair 1 (U1 vs T1): trained-only 15, untuned-only 34, by backend

| backend (task counts under every class it involves) | tasks | trained only | untuned only | net |
|---|---|---|---|---|
| GorillaFileSystem | 50 | 5 | 9 | -4 |
| MathAPI | 25 | 1 | 3 | -2 |
| MessageAPI | 40 | 3 | 8 | -5 |
| TicketAPI | 31 | 4 | 7 | -3 |
| TradingBot | 50 | 3 | 2 | +1 |
| TravelAPI | 50 | 7 | 9 | -2 |
| TwitterAPI | 39 | 3 | 8 | -5 |
| VehicleControlAPI | 50 | 0 | 14 | -14 |

| exact class set | tasks | trained only | untuned only |
|---|---|---|---|
| TwitterAPI + VehicleControlAPI | 14 | 0 | 5 |
| GorillaFileSystem | 13 | 1 | 3 |
| TravelAPI | 13 | 2 | 2 |
| TravelAPI + TwitterAPI | 10 | 2 | 2 |
| TicketAPI + TravelAPI | 15 | 2 | 2 |
| MessageAPI + TravelAPI | 10 | 1 | 3 |
| GorillaFileSystem + TicketAPI | 5 | 1 | 2 |
| GorillaFileSystem + MessageAPI | 8 | 1 | 2 |
| VehicleControlAPI | 19 | 0 | 3 |
| GorillaFileSystem + TwitterAPI | 12 | 1 | 1 |
| GorillaFileSystem + MathAPI | 12 | 1 | 1 |
| MessageAPI + VehicleControlAPI | 9 | 0 | 2 |
| TicketAPI + VehicleControlAPI | 2 | 0 | 2 |
| MathAPI + VehicleControlAPI | 6 | 0 | 2 |
| MessageAPI + TradingBot | 13 | 1 | 1 |
| TicketAPI + TradingBot | 9 | 1 | 1 |
| TradingBot | 20 | 1 | 0 |

## C. pair 2 (U2 vs T2): trained-only 15, untuned-only 35, by backend

| backend (task counts under every class it involves) | tasks | trained only | untuned only | net |
|---|---|---|---|---|
| GorillaFileSystem | 50 | 5 | 9 | -4 |
| MathAPI | 25 | 1 | 3 | -2 |
| MessageAPI | 40 | 3 | 8 | -5 |
| TicketAPI | 31 | 4 | 8 | -4 |
| TradingBot | 50 | 3 | 3 | +0 |
| TravelAPI | 50 | 7 | 9 | -2 |
| TwitterAPI | 39 | 3 | 8 | -5 |
| VehicleControlAPI | 50 | 0 | 14 | -14 |

| exact class set | tasks | trained only | untuned only |
|---|---|---|---|
| TwitterAPI + VehicleControlAPI | 14 | 0 | 5 |
| GorillaFileSystem | 13 | 1 | 3 |
| TravelAPI | 13 | 2 | 2 |
| TravelAPI + TwitterAPI | 10 | 2 | 2 |
| TicketAPI + TravelAPI | 15 | 2 | 2 |
| MessageAPI + TravelAPI | 10 | 1 | 3 |
| GorillaFileSystem + TicketAPI | 5 | 1 | 2 |
| GorillaFileSystem + MessageAPI | 8 | 1 | 2 |
| VehicleControlAPI | 19 | 0 | 3 |
| TicketAPI + TradingBot | 9 | 1 | 2 |
| GorillaFileSystem + TwitterAPI | 12 | 1 | 1 |
| GorillaFileSystem + MathAPI | 12 | 1 | 1 |
| MessageAPI + VehicleControlAPI | 9 | 0 | 2 |
| TicketAPI + VehicleControlAPI | 2 | 0 | 2 |
| MathAPI + VehicleControlAPI | 6 | 0 | 2 |
| MessageAPI + TradingBot | 13 | 1 | 1 |
| TradingBot | 20 | 1 | 0 |

## D. Per task means over the 200 tasks (tool calls = call strings in the stored result; API calls and generated tokens from multi_turn_timings)

| run | mean tool calls | mean API calls | mean generated tokens | mean prompt tokens (summed over requests) | mean generated tokens, passed tasks | mean generated tokens, failed tasks |
|---|---|---|---|---|---|---|
| U1 | 6.42 | 9.04 | 2521 | 17833 | 2281 (88 tasks) | 2711 (112 tasks) |
| T1 | 6.54 | 9.52 | 343 | 9773 | 325 (69 tasks) | 352 (131 tasks) |
| T2 | 6.54 | 9.51 | 342 | 9682 | 325 (69 tasks) | 352 (131 tasks) |
| U2 | 6.42 | 9.04 | 2514 | 17777 | 2284 (89 tasks) | 2699 (111 tasks) |

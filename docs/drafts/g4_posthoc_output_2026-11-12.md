# G4 post-hoc reading (exploratory; written after the registered analysis; decides nothing)

Runs read: U1, T1, T2, U2 from `20261111_203247_g4run.JoHtbc`. Passes are the harness's own pass list (errored tasks already counted as fails).

### Passes by task-id block

| group | tasks | U1 | T1 | T2 | U2 |
|---|---|---|---|---|---|
| ids 0-49 | 50 | 10 | 6 | 6 | 10 |
| ids 50-99 | 50 | 17 | 3 | 3 | 17 |
| ids 100-149 | 50 | 40 | 41 | 41 | 41 |
| ids 150-199 | 50 | 21 | 19 | 19 | 21 |

### Passes by BFCL tool class (a task counts under every class it involves)

| group | tasks | U1 | T1 | T2 | U2 |
|---|---|---|---|---|---|
| GorillaFileSystem | 50 | 10 | 6 | 6 | 10 |
| MathAPI | 25 | 9 | 7 | 7 | 9 |
| MessageAPI | 40 | 20 | 15 | 15 | 20 |
| TicketAPI | 31 | 15 | 12 | 12 | 16 |
| TradingBot | 50 | 40 | 41 | 41 | 41 |
| TravelAPI | 50 | 21 | 19 | 19 | 21 |
| TwitterAPI | 39 | 12 | 7 | 7 | 12 |
| VehicleControlAPI | 50 | 17 | 3 | 3 | 17 |

### Passes by exact class set

| group | tasks | U1 | T1 | T2 | U2 |
|---|---|---|---|---|---|
| TradingBot | 20 | 16 | 17 | 17 | 16 |
| VehicleControlAPI | 19 | 5 | 2 | 2 | 5 |
| TicketAPI + TravelAPI | 15 | 4 | 4 | 4 | 4 |
| TwitterAPI + VehicleControlAPI | 14 | 5 | 0 | 0 | 5 |
| GorillaFileSystem | 13 | 3 | 1 | 1 | 3 |
| MessageAPI + TradingBot | 13 | 10 | 10 | 10 | 10 |
| TravelAPI | 13 | 8 | 8 | 8 | 8 |
| GorillaFileSystem + TwitterAPI | 12 | 2 | 2 | 2 | 2 |
| GorillaFileSystem + MathAPI | 12 | 1 | 1 | 1 | 1 |
| TravelAPI + TwitterAPI | 10 | 3 | 3 | 3 | 3 |
| MessageAPI + TravelAPI | 10 | 5 | 3 | 3 | 5 |
| MessageAPI + VehicleControlAPI | 9 | 3 | 1 | 1 | 3 |
| TicketAPI + TradingBot | 9 | 7 | 7 | 7 | 8 |
| GorillaFileSystem + MessageAPI | 8 | 2 | 1 | 1 | 2 |
| MathAPI + VehicleControlAPI | 6 | 2 | 0 | 0 | 2 |
| GorillaFileSystem + TicketAPI | 5 | 2 | 1 | 1 | 2 |
| MathAPI + TradingBot | 5 | 5 | 5 | 5 | 5 |
| TradingBot + TwitterAPI | 3 | 2 | 2 | 2 | 2 |
| TicketAPI + VehicleControlAPI | 2 | 2 | 0 | 0 | 2 |
| MathAPI + TravelAPI | 2 | 1 | 1 | 1 | 1 |

### Failure messages per run (checker message, normalised; one primary message per failed task)

| failure message | U1 | T1 | T2 | U2 |
|---|---|---|---|---|
| under-call (required call missing from the execution results) | 46 | 41 | 40 | 46 |
| state mismatch: VehicleControlAPI | 19 | 32 | 33 | 19 |
| state mismatch: GorillaFileSystem | 15 | 20 | 20 | 15 |
| state mismatch: TravelAPI | 9 | 17 | 17 | 9 |
| empty turn (model response list is empty) | 12 | 0 | 0 | 12 |
| state mismatch: TwitterAPI | 4 | 7 | 7 | 4 |
| state mismatch: MessageAPI | 2 | 8 | 8 | 2 |
| state mismatch: TicketAPI | 5 | 5 | 5 | 4 |
| state mismatch: TradingBot | 0 | 1 | 1 | 0 |

Note: the table counts checker messages, and a task can carry more than one; wrong-call, extra-call and format failures are not separable from these messages alone (the checker reports state or missing calls, not why). A finer breakdown would need the stored replies (task text), which this script does not read.

### Training coverage (frozen rows' metadata only: `exercised` classes per row)

| class | training rows | training tasks | BFCL 200 tasks involving it |
|---|---|---|---|
| GorillaFileSystem | 1132 | 94 | 50 |
| MathAPI | 512 | 50 | 25 |
| MessageAPI | 159 | 15 | 40 |
| TicketAPI | 672 | 80 | 31 |
| TradingBot | 1056 | 124 | 50 |
| TravelAPI | 125 | 17 | 50 |
| TwitterAPI | 424 | 51 | 39 |
| VehicleControlAPI | 240 | 35 | 50 |

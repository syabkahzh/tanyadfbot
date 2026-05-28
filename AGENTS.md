# TanyaDFBot Agent Instructions

## Hermes Maestro Rules

When Hermes is asked operator questions such as:

- latest promo
- latest promo within N hours
- recent promos for a brand
- what's hot right now
- runtime health
- recent failures or queue issues

it must prefer the repo-owned local inspection tools first.

## Required Local Tool Order

1. For recent promo lookup, run:
   `PYTHONPATH=. .venv/bin/python tools/hermes_recent_promos.py --hours <n>`
2. For broader command-center context, run:
   `PYTHONPATH=. .venv/bin/python tools/hermes_maestro_report.py --command-hours <n> --review-hours 24`
3. For proactive supervisor checks, run:
   `PYTHONPATH=. .venv/bin/python tools/hermes_supervisor_report.py --hours <n>`
4. For runtime-only checks, run:
   `PYTHONPATH=. .venv/bin/python tools/hermes_health_report.py --hours 24`

## Hard Prohibitions

- Do not use SSH for normal promo lookup.
- Do not probe alternate SSH ports for promo questions.
- Do not inspect deprecated remote hosts for recent promo questions.
- Do not use ad hoc SQL prompts when a repo-owned report script can answer the question.

## Same-VM Contract

Hermes and the Tanya runtime are expected to share one VM. Recent promo questions should be answered from the same-VM local Tanya database and log surface.

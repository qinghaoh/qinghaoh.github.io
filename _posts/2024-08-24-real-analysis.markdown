---
title:  "Real Analysis"
category: math
tags: math
mermaid: true
---

## Enum

```mermaid
flowchart LR
    ZF@{ shape: circle, label: "Zermelo–Fraenkel axioms" }
    PP@{ shape: circle, label: "Peano Postulates" }
    N[Natural numbers]
    OpA[Addition]
    OpM[Multiplication]
    WOP[Well-Ordering Principle]

    ZF --> PP
    PP -- Unique --> N
    PP -- Definition by Recursion --> OpA
    PP -- Definition by Recursion --> OpM
    N -.-> WOP
```

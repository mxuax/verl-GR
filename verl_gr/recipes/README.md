# `verl_gr/recipes`

Generative recommendation systems (GenRecSys) use LLMs to produce recommendation rankings. 
Technically, their approaches are two-fold: output tokens can be treated either as semantic IDs (SIDs) embedded for products, goods, or items, or as natural-language representations of ranked items.

 `verl-gr` currently supports three recipes for GRPO training:
* [OpenOneRec](openonerec/README.md)
* [MiniOneRec](minionerec/README.md)
* [Rank-GRPO](rankgrpo/README.md)

We picked these three works for the initial release as they cover the two major routes of GenRecSys, and their intersection.

```
+------------------------------------------------------------------+
| OpenOneRec GRPO recipe                                           |
| user history -> policy LLM                                       |
|              -> stage 1: natural-language thinking context       |
|              -> stage 2: beam-search SID generation              |
|              -> beam-width SIDs as ranking results               |
|                 |                                                |
|                 +-- intersection of NL thinking and SID output   |
+------------------------------------------------------------------+
```

```
+------------------------------------------------------------------+
| MiniOneRec GRPO recipe                                           |
| user history -> policy LLM                                       |
|              -> beam-search SID generation                       |
|              -> beam-width SIDs as ranking results               |
|                 |                                                |
|                 +-- SID route                                    |
+------------------------------------------------------------------+
```

```
+------------------------------------------------------------------+
| Rank-GRPO recipe                                                 |
| user request -> policy LLM                                       |
|              -> natural-language ranked items                    |
|                 |                                                |
|                 +-- natural-language ranking route               |
+------------------------------------------------------------------+
```
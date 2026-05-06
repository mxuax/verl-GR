"""Rank-GRPO recipe components for verl-gr."""

__all__ = ["RankGRPODataset", "RankGRPOTask", "collate_fn", "compute_score"]


def __getattr__(name: str):
    if name in {"RankGRPODataset", "collate_fn"}:
        from verl_gr.recipes.rankgrpo.rankgrpo_dataset import RankGRPODataset, collate_fn

        return {"RankGRPODataset": RankGRPODataset, "collate_fn": collate_fn}[name]
    if name == "RankGRPOTask":
        from verl_gr.recipes.rankgrpo.rankgrpo_task import RankGRPOTask

        return RankGRPOTask
    if name == "compute_score":
        from verl_gr.recipes.rankgrpo.rankgrpo_reward import compute_score

        return compute_score
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


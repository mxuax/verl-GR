"""SFT training entry point for verl-GR.

Wraps verl's ``SFTTrainer`` with a Hydra entry point that locates the
upstream ``sft_trainer_engine.yaml`` config via ``importlib.resources``.

Usage::

    torchrun --standalone --nnodes=1 --nproc_per_node=4 \\
        -m verl_gr.trainers.sft_trainer \\
        data.train_files=/path/to/train.parquet \\
        data.custom_cls.path=verl_gr.recipes.minionerec.data.sft_dataset \\
        data.custom_cls.name=MiniOneRecSFTDataset \\
        model.path=/path/to/model \\
        ...
"""

from __future__ import annotations


def run_sft(config):
    from verl.trainer.sft_trainer import SFTTrainer
    from verl.utils.distributed import (
        destroy_global_process_group,
        initialize_global_process_group,
    )

    initialize_global_process_group()
    trainer = SFTTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


def main():
    import importlib.resources

    import hydra
    from verl.utils.device import auto_set_device

    traversable = importlib.resources.files("verl.trainer.config")
    config_dir = str(traversable)

    @hydra.main(config_path=config_dir, config_name="sft_trainer_engine", version_base=None)
    def _main(config):
        auto_set_device(config)
        run_sft(config)

    return _main()


if __name__ == "__main__":
    main()

import argparse
import os
from collections import defaultdict
from typing import Any, Iterable, Optional

import shtab
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI


def infer_metric_mode(metric: str) -> str:
    lower_is_better = ["loss"]
    metric = metric.lower()
    for m in lower_is_better:
        if m in metric:
            return "min"
    return "max"


class LitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("-n", "--name", default=None, help="Experiment name")
        parser.add_argument("-m", "--monitor", type=str, help="Monitor in EarlyStop and ModelCheckpoint callbacks")
        parser.add_argument(
            "-d",
            "--debug",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Debug mode",
        )

    def before_instantiate_classes(self) -> None:
        config = self.config[self.subcommand]
        mode = "debug" if config.debug else self.subcommand

        config.trainer.default_root_dir = os.path.join("results", mode)

        if config.debug:
            self.save_config_callback = None
            config.trainer.logger = None

        logger = config.trainer.logger
        assert logger is not True, "should assign trainer.logger with the specific logger."
        if logger:
            loggers = logger if isinstance(logger, Iterable) else [logger]
            for logger in loggers:
                logger.init_args.save_dir = os.path.join(
                    logger.init_args.get("save_dir", "results"), self.subcommand
                )
                if config.name:
                    logger.init_args.name = config.name

        for cb in config.trainer.callbacks:
            if cb.class_path.split(".")[-1] in ["EarlyStopping", "ModelCheckpoint"]:
                monitor: Optional[str] = config.get("monitor", None)
                assert monitor is not None, \
                    "When mode in callback is set to be auto, monitor must be defined!"

                cb.init_args.mode = infer_metric_mode(monitor)
                cb.init_args.monitor = monitor

    def setup_parser(
            self,
            add_subcommands: bool,
            main_kwargs: dict[str, Any],
            subparser_kwargs: dict[str, Any],
    ) -> None:
        """Initialize and setup the parser, subcommands, and arguments."""
        # move default_config_files to subparser_kwargs
        if add_subcommands:
            default_configs = main_kwargs.pop("default_config_files", None)
            subparser_kwargs = defaultdict(dict, subparser_kwargs)
            for subcmd in self.subcommands():
                subparser_kwargs[subcmd]["default_config_files"] = default_configs

        self.parser = self.init_parser(**main_kwargs)
        shtab.add_argument_to(self.parser, ["-s", "--print-completion"])

        if add_subcommands:
            self._subcommand_method_arguments: dict[str, list[str]] = {}
            self._add_subcommands(self.parser, **subparser_kwargs)
        else:
            self._add_arguments(self.parser)

"""Public five-fold training entrypoint for PoMoG."""

from __future__ import annotations

from pomog.train.train_pomog import _build_parser, load_config
from pomog.train.train_pomog import DEFAULT_CONFIG, train_pomog

__all__ = ["main", "train_pomog"]


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config(DEFAULT_CONFIG, args.config)
    train_pomog(args, config=config)


if __name__ == "__main__":
    main()

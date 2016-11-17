# -*- coding: utf-8 -*-
import pycrfsuite
from tabulate import tabulate


class LinePerIterationTrainer(pycrfsuite.Trainer):
    """
    This pycrfsuite.Trainer prints information about each iteration
    on a single line.
    """
    def on_iteration(self, log, info):
        parts = [
            "Iter {num:<3} ",
            "time={time:<5.2f} ",
            "loss={loss:<8.2f} ",
        ]

        if 'active_features' in info:
            parts += ["active={active_features:<5} "]

        if 'avg_precision' in info:
            parts += [
                "precision={avg_precision:0.3f}  ",
                "recall={avg_recall:0.3f}  ",
                "F1={avg_f1:0.3f}  ",
                "Acc(item/seq)={item_accuracy_float:0.3f} {instance_accuracy_float:0.3f}  "
            ]

        if 'feature_norm' in info:
            parts += ["feature_norm={feature_norm:<8.2f}"]

        line = "".join(parts)
        print(line.format(**info).strip())

    def on_optimization_end(self, log):
        last_iter = self.logparser.last_iteration
        if last_iter.get('scores', None):
            data = [
                [entity, score.precision, score.recall, score.f1 or 0, score.ref]
                for entity, score in sorted(last_iter['scores'].items())
            ]
            table = tabulate(data,
                headers=["Label", "Precision", "Recall", "F1", "Support"],
                floatfmt="0.3f",
            )
            size = len(table.splitlines()[0])
            print("="*size)
            print(table)
            print("-"*size)
        super(LinePerIterationTrainer, self).on_optimization_end(log)

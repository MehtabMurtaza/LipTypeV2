from __future__ import annotations

import typer

from liptype_rebuild.cli import preprocess as preprocess_cmd
from liptype_rebuild.cli import train as train_cmd
from liptype_rebuild.cli import predict as predict_cmd
from liptype_rebuild.cli import enhance as enhance_cmd
from liptype_rebuild.cli import eval as eval_cmd

app = typer.Typer(help="LipType TF2 rebuild CLI.")

app.add_typer(preprocess_cmd.app, name="preprocess")
app.add_typer(enhance_cmd.app, name="enhance")
app.add_typer(train_cmd.app, name="train")
app.add_typer(eval_cmd.app, name="eval")
app.add_typer(predict_cmd.app, name="predict")


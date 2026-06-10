from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .config import get_settings
from .graph import build_graph
from .utils import ensure_dir, make_project_id, save_json


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(prompt: str, project_id: str | None = None) -> None:
    settings = get_settings()
    project_id = project_id or make_project_id()
    project_dir = ensure_dir(Path(settings.output_dir) / project_id)

    console.print(f"[bold]Project:[/bold] {project_id}")
    console.print(f"[bold]Output:[/bold] {project_dir.resolve()}")
    console.print("[bold]Running LangGraph workflow...[/bold]")

    graph = build_graph(settings)
    initial_state = {
        "project_id": project_id,
        "user_prompt": prompt,
        "output_dir": str(project_dir),
        "errors": [],
    }
    final_state = graph.invoke(initial_state)
    save_json(project_dir / "state.json", final_state)

    console.print("[green]Done.[/green]")
    console.print(f"Script: {project_dir / 'script.json'}")
    for path in final_state.get("character_images", []):
        console.print(f"Character image: {path}")
    for path in final_state.get("scene_images", []):
        console.print(f"Scene image: {path}")
    for path in final_state.get("animatic_videos", []):
        console.print(f"Animatic: {path}")
    for path in final_state.get("shot_videos", []):
        console.print(f"Shot video: {path}")
    if final_state.get("mixed_audio"):
        console.print(f"Mixed audio: {final_state['mixed_audio']}")
    if final_state.get("final_video"):
        console.print(f"[bold green]Final video:[/bold green] {final_state['final_video']}")


if __name__ == "__main__":
    app()

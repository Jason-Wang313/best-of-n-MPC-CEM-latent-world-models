from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "iclr_submission"
DESKTOP = Path.home() / "OneDrive" / "Desktop"
if not DESKTOP.exists():
    DESKTOP = Path.home() / "Desktop"
DESKTOP_PDF = DESKTOP / "best-of-n-MPC-CEM-latent-world-models-v3.pdf"
FINAL_PDF = ROOT / "paper" / "final" / "best-of-n-MPC-CEM-latent-world-models-v3.pdf"


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, text=True)


def clean_latex() -> None:
    for suffix in (".aux", ".bbl", ".blg", ".log", ".out", ".toc", ".pdf"):
        path = PAPER / f"main{suffix}"
        if path.exists():
            path.unlink()


def build_latex() -> Path:
    clean_latex()
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], PAPER)
    run(["bibtex", "main"], PAPER)
    for _ in range(4):
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], PAPER)
    pdf = PAPER / "main.pdf"
    if not pdf.exists():
        raise FileNotFoundError(pdf)
    return pdf


def main() -> None:
    run(["python", str(ROOT / "experiments" / "v3_cached_evidence.py")], ROOT)
    pdf = build_latex()
    FINAL_PDF.parent.mkdir(parents=True, exist_ok=True)
    DESKTOP_PDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, FINAL_PDF)
    shutil.copy2(pdf, DESKTOP_PDF)
    print(f"PDF: {DESKTOP_PDF}")
    print(f"Repo PDF: {FINAL_PDF}")


if __name__ == "__main__":
    main()

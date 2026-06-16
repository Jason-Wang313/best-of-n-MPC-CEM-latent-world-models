from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "iclr_submission"
DESKTOP = Path.home() / "OneDrive" / "Desktop"
if not DESKTOP.exists():
    DESKTOP = Path.home() / "Desktop"
DESKTOP_PDF = DESKTOP / "best-of-n-MPC-CEM-latent-world-models-v4.pdf"
FINAL_PDF = ROOT / "paper" / "final" / "best-of-n-MPC-CEM-latent-world-models-v4.pdf"
OLD_DESKTOP_PDF = DESKTOP / "best-of-n-MPC-CEM-latent-world-models-v3.pdf"
OLD_FINAL_PDF = ROOT / "paper" / "final" / "best-of-n-MPC-CEM-latent-world-models-v3.pdf"


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
    run(["python", str(ROOT / "experiments" / "v4_protocol_evidence.py")], ROOT)
    run(["python", str(ROOT / "experiments" / "19_v4_gymnasium_cem_cards.py")], ROOT)
    pdf = build_latex()
    FINAL_PDF.parent.mkdir(parents=True, exist_ok=True)
    DESKTOP_PDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, FINAL_PDF)
    shutil.copy2(pdf, DESKTOP_PDF)
    for old_pdf in (OLD_DESKTOP_PDF, OLD_FINAL_PDF):
        if old_pdf.exists():
            old_pdf.unlink()
    print(f"PDF: {DESKTOP_PDF}")
    print(f"Repo PDF: {FINAL_PDF}")


if __name__ == "__main__":
    main()

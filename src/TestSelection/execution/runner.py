from __future__ import annotations

import json
from pathlib import Path


class ExecutionRunner:
    """Orchestrates Robot Framework execution with test selection.

    Builds the appropriate robot CLI arguments based on whether
    the selection includes DataDriver tests.
    """

    def __init__(
        self,
        suite_path: str | Path,
        selection_file: str | Path,
        output_dir: str | Path = "./results",
    ) -> None:
        self._suite_path = Path(suite_path)
        self._selection_file = Path(selection_file)
        self._output_dir = Path(output_dir)
        self._selection_data = json.loads(self._selection_file.read_text())
        self._has_datadriver = any(
            t.get("is_datadriver", False)
            for t in self._selection_data["selected"]
        )

    def build_robot_args(self) -> list[str]:
        """Build robot CLI arguments with --prerunmodifier and --listener as needed."""
        args: list[str] = [
            "--outputdir",
            str(self._output_dir),
            "--prerunmodifier",
            f"TestSelection.execution.prerun_modifier.DiversePreRunModifier:{self._selection_file}",
        ]
        if self._has_datadriver:
            args.extend([
                "--listener",
                f"TestSelection.execution.listener.DiverseDataDriverListener:{self._selection_file}",
            ])
        return args

    def execute(self, extra_args: list[str] | None = None) -> int:
        """Run robot.run_cli with built arguments. Returns exit code."""
        import robot

        args = self.build_robot_args()
        if extra_args:
            args.extend(extra_args)
        args.append(str(self._suite_path))
        return robot.run_cli(args, exit=False)  # type: ignore[attr-defined]

    def generate_report(self, return_code: int) -> dict:
        """Generate selection_report.json with execution metadata."""
        selected_count = len(self._selection_data["selected"])
        dd_count = sum(
            1
            for t in self._selection_data["selected"]
            if t.get("is_datadriver", False)
        )
        report = {
            "return_code": return_code,
            "suite_path": str(self._suite_path),
            "selection_file": str(self._selection_file),
            "selected_tests": selected_count,
            "datadriver_tests": dd_count,
            "standard_tests": selected_count - dd_count,
            "strategy": self._selection_data.get("strategy", "unknown"),
            "status": "pass" if return_code == 0 else "fail",
        }
        self._output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self._output_dir / "selection_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        return report

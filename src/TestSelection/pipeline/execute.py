"""Stage 3 orchestrator: execute selected tests via Robot Framework."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_execute(
    suite_path: Path,
    selection_file: Path,
    output_dir: str = "./results",
    extra_robot_args: list[str] | None = None,
) -> int:
    """Run the execution stage.

    Returns the Robot Framework exit code (0=pass, 1=fail, 2=error).
    """
    try:
        from TestSelection.execution.runner import ExecutionRunner

        runner = ExecutionRunner(
            suite_path=suite_path,
            selection_file=selection_file,
            output_dir=output_dir,
        )

        logger.info(
            "[DIVERSE-SELECT] stage=execute event=start "
            "suite=%s selection=%s",
            suite_path,
            selection_file,
        )

        return_code = runner.execute(extra_args=extra_robot_args)
        runner.generate_report(return_code)

        logger.info(
            "[DIVERSE-SELECT] stage=execute event=complete "
            "return_code=%d",
            return_code,
        )

        return return_code

    except Exception as exc:
        logger.warning(
            "[DIVERSE-SELECT] stage=execute event=error error=%s",
            str(exc),
        )
        return 2

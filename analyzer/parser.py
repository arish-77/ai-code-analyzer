import ast
from dataclasses import dataclass, field




# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    type: str
    line: int
    message: str


@dataclass
class AnalysisResult:
    issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "issues": [
                {"type": i.type, "line": i.line, "message": i.message}
                for i in self.issues
            ]
        }


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def detect_unused_variables(tree: ast.Module) -> list[Issue]:
    issues = []

    def analyze_scope(node):
        assigned = {}
        used = set()

        # ✅ Step 1: collect assigned variables (only direct level)
        for child in ast.iter_child_nodes(node):

            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        assigned[target.id] = target.lineno

            elif isinstance(child, ast.AnnAssign):
                if isinstance(child.target, ast.Name) and child.value:
                    assigned[child.target.id] = child.target.lineno

            elif isinstance(child, ast.For):
                if isinstance(child.target, ast.Name):
                    assigned[child.target.id] = child.target.lineno

        # ✅ Step 2: collect ALL used variables (deep traversal)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                used.add(sub.id)

        # ✅ Step 3: compare
        for name, lineno in assigned.items():
            if name not in used and not name.startswith("_"):
                issues.append(Issue(
                    type="unused_variable",
                    line=lineno,
                    message=f"Variable '{name}' is assigned but never used.",
                ))

    # Analyze each function
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            analyze_scope(node)

    return issues


def detect_long_functions(tree: ast.Module, max_lines: int = 20) -> list[Issue]:
    """
    Flags any function/method whose body spans more than `max_lines` lines.
    Line count = last line of body − first line of def statement.
    """
    issues = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        start = node.lineno
        end   = max(
            getattr(child, "end_lineno", start)
            for child in ast.walk(node)
        )
        length = end - start + 1

        if length > max_lines:
            issues.append(Issue(
                type="long_function",
                line=start,
                message=(
                    f"Function '{node.name}' is {length} lines long "
                    f"(limit: {max_lines})."
                ),
            ))

    return issues


def detect_deep_nesting(tree: ast.Module, max_depth: int = 3) -> list[Issue]:
    """
    Flags block-level statements whose nesting depth exceeds `max_depth`.
    Nesting is counted for: if / for / while / with / try blocks.
    """
    NESTING_NODES = (ast.If, ast.For, ast.While, ast.With, ast.Try)
    issues: list[Issue] = []
    seen_lines: set[int] = set()   # avoid duplicate reports on the same line

    def walk_depth(node: ast.AST, depth: int) -> None:
        if isinstance(node, NESTING_NODES):
            depth += 1
            if depth > max_depth:
                lineno = getattr(node, "lineno", 0)
                if lineno not in seen_lines:
                    seen_lines.add(lineno)
                    issues.append(Issue(
                        type="deep_nesting",
                        line=lineno,
                        message=(
                            f"Nesting depth of {depth} exceeds "
                            f"the limit of {max_depth}."
                        ),
                    ))

        for child in ast.iter_child_nodes(node):
            walk_depth(child, depth)

    walk_depth(tree, depth=0)
    return issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(source_code: str) -> dict:
    """
    Parse `source_code` and run all detectors.

    Returns:
        {
            "issues": [
                {"type": str, "line": int, "message": str},
                ...
            ]
        }
    Raises:
        SyntaxError: if the source code cannot be parsed.
    """
    tree = ast.parse(source_code)

    result = AnalysisResult()
    result.issues += detect_unused_variables(tree)
    result.issues += detect_long_functions(tree)
    result.issues += detect_deep_nesting(tree)

    # Sort by line number for a readable report
    # Remove duplicates
    seen = set()
    unique_issues = []

    for issue in result.issues:
        key = (issue.type, issue.line, issue.message)
        if key not in seen:
            seen.add(key)
            unique_issues.append(issue)

    # Sort clean issues
    unique_issues.sort(key=lambda i: i.line)

    # Keep only one issue per type (best / most severe)
    best_issues = {}

    for issue in unique_issues:
        if issue.type not in best_issues:
            best_issues[issue.type] = issue
        else:
            # pick the "worse" one (higher line number OR deeper issue)
            if issue.line > best_issues[issue.type].line:
                best_issues[issue.type] = issue

    result.issues = list(best_issues.values())

    # Sort again
    result.issues.sort(key=lambda i: i.line)

    return result.to_dict()
    
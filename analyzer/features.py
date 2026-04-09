import ast


def extract_features(source_code: str) -> dict:
    """
    Parse source_code and extract three structural features.

    Returns:
        {
            "number_of_functions":    int,
            "average_function_length": float,
            "max_nesting_depth":       int,
        }

    Raises:
        SyntaxError: if source_code is not valid Python.
    """
    tree = ast.parse(source_code)

    return {
        "number_of_functions":     _count_functions(tree),
        "average_function_length": _average_function_length(tree),
        "max_nesting_depth":       _max_nesting_depth(tree),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _count_functions(tree: ast.Module) -> int:
    """Count every function and async function definition in the file."""
    return sum(
        1 for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _average_function_length(tree: ast.Module) -> float:
    """
    Average line-span of all functions.
    span = (last line of body) - (def line) + 1
    Returns 0.0 when there are no functions.
    """
    lengths = [
        _function_line_span(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return round(sum(lengths) / len(lengths), 2) if lengths else 0.0


def _function_line_span(func_node: ast.FunctionDef) -> int:
    """Return the line count of a single function node."""
    last_line = max(
        getattr(child, "end_lineno", func_node.lineno)
        for child in ast.walk(func_node)
    )
    return last_line - func_node.lineno + 1


def _max_nesting_depth(tree: ast.Module) -> int:
    """
    Walk the AST recursively, tracking depth through block-level nodes.
    Returns the deepest nesting level found anywhere in the file.
    """
    BLOCK_NODES = (ast.If, ast.For, ast.While, ast.With, ast.Try)

    def _walk(node: ast.AST, current_depth: int) -> int:
        if isinstance(node, BLOCK_NODES):
            current_depth += 1

        child_max = max(
            (_walk(child, current_depth) for child in ast.iter_child_nodes(node)),
            default=current_depth,
        )
        return max(current_depth, child_max)

    return _walk(tree, 0)
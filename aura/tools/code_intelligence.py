"""Code Intelligence Tool: project-wide symbol extraction, call graphs,
type analysis, and smart navigation.

Uses Python's `ast` module for Python files and regex for JS/TS.
No external dependencies (no tree-sitter, no networkx required).

Capabilities:
  - Deep symbol extraction (functions, classes, imports, exports, constants)
  - Call graph analysis (callers, callees, dependency chains)
  - Type annotation extraction (Python)
  - Smart navigation (find_definition, find_references, get_context, get_dependencies)
  - Language support: Python (ast), JavaScript/TypeScript (regex)
"""

import ast
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "coverage", ".pytest_cache",
    ".mypy_cache", ".tox", ".eggs", ".cache", ".parcel-cache",
    ".turbo", ".svelte-kit", "target", "vendor", ".gradle",
    ".idea", ".vs", ".vscode",
})

EXT_LANG = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
}

_BLOCKED_ROOTS = {
    "/", "/bin", "/sbin", "/usr", "/etc", "/var", "/proc", "/sys",
    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
}

MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class SymbolInfo:
    """Rich representation of a code symbol."""

    __slots__ = (
        "file", "name", "kind", "line", "end_line", "sig",
        "docstring", "decorators", "bases", "methods", "properties",
        "params", "return_type", "is_async", "is_exported",
        "body_names_called",
    )

    def __init__(self, **kw):
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot))

    def to_dict(self) -> dict:
        d = {}
        for slot in self.__slots__:
            v = getattr(self, slot, None)
            if v is not None:
                d[slot] = v
        return d


class ImportInfo:
    """Representation of an import statement."""

    __slots__ = ("file", "line", "module", "names", "alias", "is_from", "level")

    def __init__(self, **kw):
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot))

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__ if getattr(self, s, None) is not None}


# ---------------------------------------------------------------------------
# Python AST extractor
# ---------------------------------------------------------------------------

class _PythonExtractor:
    """Extract rich symbol information from Python files using the ast module."""

    def extract(self, source: str, rel_path: str) -> Tuple[List[SymbolInfo], List[ImportInfo]]:
        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError:
            return [], []

        symbols: List[SymbolInfo] = []
        imports: List[ImportInfo] = []
        lines = source.split("\n")

        self._walk(tree, symbols, imports, rel_path, lines, parent_class=None)
        return symbols, imports

    # ------------------------------------------------------------------

    def _walk(self, node, symbols, imports, rel_path, lines, parent_class):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(self._extract_function(child, rel_path, lines, parent_class))
            elif isinstance(child, ast.ClassDef):
                cls_sym = self._extract_class(child, rel_path, lines)
                symbols.append(cls_sym)
                # Recurse into class body for methods
                self._walk(child, symbols, imports, rel_path, lines, parent_class=child.name)
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    imports.append(ImportInfo(
                        file=rel_path, line=child.lineno,
                        module=alias.name, names=[alias.name],
                        alias=alias.asname, is_from=False, level=0,
                    ))
            elif isinstance(child, ast.ImportFrom):
                mod = child.module or ""
                names = [a.name for a in child.names] if child.names else []
                imports.append(ImportInfo(
                    file=rel_path, line=child.lineno,
                    module=mod, names=names,
                    alias=None, is_from=True, level=child.level or 0,
                ))
            elif isinstance(child, ast.Assign):
                self._extract_assignment(child, symbols, rel_path, lines)
            elif isinstance(child, ast.AnnAssign):
                self._extract_ann_assign(child, symbols, rel_path, lines)

    # ------------------------------------------------------------------
    # Functions / methods
    # ------------------------------------------------------------------

    def _extract_function(self, node, rel_path, lines, parent_class) -> SymbolInfo:
        is_async = isinstance(node, ast.AsyncFunctionDef)
        params = self._extract_params(node.args)
        return_type = self._annotation_str(node.returns) if node.returns else None
        decorators = [self._decorator_str(d) for d in node.decorator_list]
        sig_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
        docstring = ast.get_docstring(node)

        # Collect names called inside function body
        called = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                called.add(self._call_name(child))

        kind = "method" if parent_class else "function"

        return SymbolInfo(
            file=rel_path,
            name=f"{parent_class}.{node.name}" if parent_class else node.name,
            kind=kind,
            line=node.lineno,
            end_line=node.end_lineno,
            sig=sig_line,
            docstring=docstring[:300] if docstring else None,
            decorators=decorators or None,
            params=params or None,
            return_type=return_type,
            is_async=is_async or None,
            body_names_called=sorted(called - {None, ""}) or None,
        )

    def _extract_params(self, args: ast.arguments) -> List[Dict[str, str]]:
        params = []
        # positional + normal args
        all_args = args.posonlyargs + args.args + args.kwonlyargs
        defaults_offset = len(all_args) - len(args.defaults)
        for i, arg in enumerate(all_args):
            p: Dict[str, Any] = {"name": arg.arg}
            if arg.annotation:
                p["type"] = self._annotation_str(arg.annotation)
            # defaults for positional args
            di = i - defaults_offset
            if 0 <= di < len(args.defaults):
                p["default"] = self._const_str(args.defaults[di])
            params.append(p)
        if args.vararg:
            p = {"name": f"*{args.vararg.arg}"}
            if args.vararg.annotation:
                p["type"] = self._annotation_str(args.vararg.annotation)
            params.append(p)
        if args.kwarg:
            p = {"name": f"**{args.kwarg.arg}"}
            if args.kwarg.annotation:
                p["type"] = self._annotation_str(args.kwarg.annotation)
            params.append(p)
        return params

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------

    def _extract_class(self, node: ast.ClassDef, rel_path, lines) -> SymbolInfo:
        bases = [self._annotation_str(b) for b in node.bases]
        decorators = [self._decorator_str(d) for d in node.decorator_list]
        docstring = ast.get_docstring(node)
        sig_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""

        # Extract method names and property names
        methods = []
        properties = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_property = any(
                    self._decorator_str(d) in ("property", "cached_property")
                    for d in child.decorator_list
                )
                if is_property:
                    properties.append(child.name)
                else:
                    methods.append(child.name)

        return SymbolInfo(
            file=rel_path,
            name=node.name,
            kind="class",
            line=node.lineno,
            end_line=node.end_lineno,
            sig=sig_line,
            docstring=docstring[:300] if docstring else None,
            decorators=decorators or None,
            bases=bases or None,
            methods=methods or None,
            properties=properties or None,
        )

    # ------------------------------------------------------------------
    # Assignments (globals, constants)
    # ------------------------------------------------------------------

    def _extract_assignment(self, node: ast.Assign, symbols, rel_path, lines):
        """Extract module-level variable/constant assignments."""
        # Only top-level (col_offset 0) assignments
        if getattr(node, "col_offset", -1) != 0:
            return
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                kind = "constant" if name.isupper() else "variable"
                sig_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                symbols.append(SymbolInfo(
                    file=rel_path, name=name, kind=kind,
                    line=node.lineno, sig=sig_line,
                ))

    def _extract_ann_assign(self, node: ast.AnnAssign, symbols, rel_path, lines):
        """Extract annotated assignments at module level."""
        if getattr(node, "col_offset", -1) != 0:
            return
        if isinstance(node.target, ast.Name):
            name = node.target.id
            kind = "constant" if name.isupper() else "variable"
            type_str = self._annotation_str(node.annotation)
            sig_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
            symbols.append(SymbolInfo(
                file=rel_path, name=name, kind=kind,
                line=node.lineno, sig=sig_line,
                return_type=type_str,  # reusing field for var type
            ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _annotation_str(node) -> str:
        """Convert an AST annotation node to a readable string."""
        try:
            return ast.unparse(node)
        except Exception:
            return "?"

    @staticmethod
    def _const_str(node) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return "..."

    @staticmethod
    def _decorator_str(node) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return "?"

    @staticmethod
    def _call_name(node: ast.Call) -> Optional[str]:
        """Extract the function name from a Call node."""
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None


# ---------------------------------------------------------------------------
# JS/TS regex extractor
# ---------------------------------------------------------------------------

class _JSTSExtractor:
    """Extract symbols from JavaScript/TypeScript using regex patterns."""

    # Pre-compiled patterns
    _FUNC_PATTERNS = [
        # function declarations: function foo(...) / async function foo(...)
        re.compile(
            r'^(?P<export>export\s+)?(?:default\s+)?(?P<async>async\s+)?function\s*\*?\s+(?P<name>\w+)\s*'
            r'(?:<[^>]*>\s*)?\((?P<params>[^)]*)\)(?:\s*:\s*(?P<ret>[^\s{]+))?',
            re.MULTILINE,
        ),
        # arrow / const fn: const foo = (...) => / const foo = async (...) =>
        re.compile(
            r'^(?P<export>export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*'
            r'(?::\s*[^=]+)?\s*=\s*(?P<async>async\s+)?'
            r'(?:\((?P<params>[^)]*)\)|(?P<single_param>\w+))\s*(?::\s*(?P<ret>[^\s=]+))?\s*=>',
            re.MULTILINE,
        ),
        # method in class/object: foo(...) { / async foo(...) {
        re.compile(
            r'^\s+(?P<async>async\s+)?(?P<name>\w+)\s*\((?P<params>[^)]*)\)(?:\s*:\s*(?P<ret>[^\s{]+))?\s*\{',
            re.MULTILINE,
        ),
    ]

    _CLASS_PATTERN = re.compile(
        r'^(?P<export>export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(?P<name>\w+)'
        r'(?:\s+extends\s+(?P<base>\w[\w.]*))?(?:\s+implements\s+(?P<ifaces>[^{]+))?\s*\{',
        re.MULTILINE,
    )

    _INTERFACE_PATTERN = re.compile(
        r'^(?P<export>export\s+)?interface\s+(?P<name>\w+)'
        r'(?:\s+extends\s+(?P<base>[^{]+))?\s*\{',
        re.MULTILINE,
    )

    _TYPE_ALIAS_PATTERN = re.compile(
        r'^(?P<export>export\s+)?type\s+(?P<name>\w+)\s*(?:<[^>]*>)?\s*=',
        re.MULTILINE,
    )

    _ENUM_PATTERN = re.compile(
        r'^(?P<export>export\s+)?(?:const\s+)?enum\s+(?P<name>\w+)\s*\{',
        re.MULTILINE,
    )

    # Import patterns
    _IMPORT_PATTERNS = [
        # import { a, b } from 'mod'
        re.compile(r'''import\s+\{(?P<names>[^}]+)\}\s+from\s+['"](?P<mod>[^'"]+)['"]'''),
        # import X from 'mod'
        re.compile(r'''import\s+(?P<default>\w+)\s+from\s+['"](?P<mod>[^'"]+)['"]'''),
        # import * as X from 'mod'
        re.compile(r'''import\s+\*\s+as\s+(?P<ns>\w+)\s+from\s+['"](?P<mod>[^'"]+)['"]'''),
        # import 'mod' (side-effect)
        re.compile(r'''import\s+['"](?P<mod>[^'"]+)['"]'''),
        # require('mod')
        re.compile(r'''(?:const|let|var)\s+(?:\{(?P<names>[^}]+)\}|(?P<default>\w+))\s*=\s*require\s*\(\s*['"](?P<mod>[^'"]+)['"]\s*\)'''),
    ]

    # Export patterns
    _EXPORT_PATTERNS = [
        re.compile(r'^export\s+default\s+(?:class|function|const|let|var)\s+(?P<name>\w+)', re.MULTILINE),
        re.compile(r'^export\s+\{(?P<names>[^}]+)\}', re.MULTILINE),
        re.compile(r'^export\s+(?:const|let|var|function|class|interface|type|enum)\s+(?P<name>\w+)', re.MULTILINE),
    ]

    _CONST_PATTERN = re.compile(
        r'^(?P<export>export\s+)?(?:const|let|var)\s+(?P<name>[A-Z][A-Z_0-9]+)\s*(?::\s*(?P<type>[^=]+))?\s*=',
        re.MULTILINE,
    )

    def extract(self, source: str, rel_path: str) -> Tuple[List[SymbolInfo], List[ImportInfo]]:
        symbols: List[SymbolInfo] = []
        imports: List[ImportInfo] = []

        self._extract_imports(source, rel_path, imports)
        self._extract_functions(source, rel_path, symbols)
        self._extract_classes(source, rel_path, symbols)
        self._extract_interfaces(source, rel_path, symbols)
        self._extract_type_aliases(source, rel_path, symbols)
        self._extract_enums(source, rel_path, symbols)
        self._extract_constants(source, rel_path, symbols)

        return symbols, imports

    def _line_at(self, source: str, pos: int) -> int:
        return source[:pos].count("\n") + 1

    def _extract_functions(self, source, rel_path, symbols):
        seen_lines: Set[int] = set()
        for pat in self._FUNC_PATTERNS:
            for m in pat.finditer(source):
                line = self._line_at(source, m.start())
                if line in seen_lines:
                    continue
                seen_lines.add(line)
                name = m.group("name")
                is_async = bool(m.group("async")) if "async" in m.groupdict() else False
                is_exported = bool(m.group("export")) if "export" in m.groupdict() else False
                params_str = m.group("params") if "params" in m.groupdict() and m.group("params") else ""
                ret = m.group("ret") if "ret" in m.groupdict() else None

                params = self._parse_js_params(params_str)
                symbols.append(SymbolInfo(
                    file=rel_path, name=name, kind="function",
                    line=line, sig=m.group(0).strip()[:200],
                    params=params or None,
                    return_type=ret.strip() if ret else None,
                    is_async=is_async or None,
                    is_exported=is_exported or None,
                ))

    def _extract_classes(self, source, rel_path, symbols):
        for m in self._CLASS_PATTERN.finditer(source):
            line = self._line_at(source, m.start())
            name = m.group("name")
            base = m.group("base")
            is_exported = bool(m.group("export"))
            bases = [b.strip() for b in base.split(",")] if base else None
            ifaces = m.group("ifaces")
            if ifaces:
                bases = (bases or []) + [i.strip() for i in ifaces.split(",")]

            # Find methods inside the class body
            methods = self._find_class_methods(source, m.start())

            symbols.append(SymbolInfo(
                file=rel_path, name=name, kind="class",
                line=line, sig=m.group(0).strip()[:200],
                bases=bases, methods=methods or None,
                is_exported=is_exported or None,
            ))

    def _find_class_methods(self, source: str, class_start: int) -> List[str]:
        """Find method names inside a class body by tracking braces."""
        methods = []
        # Find opening brace
        brace_pos = source.find("{", class_start)
        if brace_pos == -1:
            return methods

        depth = 1
        i = brace_pos + 1
        method_re = re.compile(r'(?:async\s+)?(\w+)\s*\([^)]*\)\s*(?::\s*[^{]+)?\s*\{')
        while i < len(source) and depth > 0:
            ch = source[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1

        class_body = source[brace_pos + 1:i - 1] if i > brace_pos + 1 else ""
        for m in method_re.finditer(class_body):
            name = m.group(1)
            if name not in ("if", "for", "while", "switch", "catch", "function"):
                methods.append(name)
        return methods

    def _extract_interfaces(self, source, rel_path, symbols):
        for m in self._INTERFACE_PATTERN.finditer(source):
            line = self._line_at(source, m.start())
            base = m.group("base")
            bases = [b.strip() for b in base.split(",")] if base else None
            symbols.append(SymbolInfo(
                file=rel_path, name=m.group("name"), kind="interface",
                line=line, sig=m.group(0).strip()[:200],
                bases=bases,
                is_exported=bool(m.group("export")) or None,
            ))

    def _extract_type_aliases(self, source, rel_path, symbols):
        for m in self._TYPE_ALIAS_PATTERN.finditer(source):
            line = self._line_at(source, m.start())
            symbols.append(SymbolInfo(
                file=rel_path, name=m.group("name"), kind="type",
                line=line, sig=m.group(0).strip()[:200],
                is_exported=bool(m.group("export")) or None,
            ))

    def _extract_enums(self, source, rel_path, symbols):
        for m in self._ENUM_PATTERN.finditer(source):
            line = self._line_at(source, m.start())
            symbols.append(SymbolInfo(
                file=rel_path, name=m.group("name"), kind="enum",
                line=line, sig=m.group(0).strip()[:200],
                is_exported=bool(m.group("export")) or None,
            ))

    def _extract_constants(self, source, rel_path, symbols):
        for m in self._CONST_PATTERN.finditer(source):
            line = self._line_at(source, m.start())
            name = m.group("name")
            type_str = m.group("type").strip() if m.group("type") else None
            symbols.append(SymbolInfo(
                file=rel_path, name=name, kind="constant",
                line=line, sig=m.group(0).strip()[:200],
                return_type=type_str,
                is_exported=bool(m.group("export")) or None,
            ))

    def _extract_imports(self, source, rel_path, imports):
        for pat in self._IMPORT_PATTERNS:
            for m in pat.finditer(source):
                gd = m.groupdict()
                mod = gd.get("mod", "")
                names = []
                if gd.get("names"):
                    names = [n.strip().split(" as ")[0].strip()
                             for n in gd["names"].split(",") if n.strip()]
                elif gd.get("default"):
                    names = [gd["default"]]
                elif gd.get("ns"):
                    names = [f"* as {gd['ns']}"]
                line = self._line_at(source, m.start())
                imports.append(ImportInfo(
                    file=rel_path, line=line, module=mod,
                    names=names, is_from=True, level=0,
                ))

    @staticmethod
    def _parse_js_params(params_str: str) -> List[Dict[str, str]]:
        if not params_str.strip():
            return []
        params = []
        for raw in params_str.split(","):
            raw = raw.strip()
            if not raw:
                continue
            # Handle destructuring
            if raw.startswith("{") or raw.startswith("["):
                params.append({"name": raw.split("}")[0] + "}" if "{" in raw else raw})
                continue
            # name: Type = default
            parts = raw.split("=", 1)
            name_type = parts[0].strip()
            p: Dict[str, str] = {}
            if ":" in name_type:
                name_part, type_part = name_type.split(":", 1)
                p["name"] = name_part.strip()
                p["type"] = type_part.strip()
            else:
                p["name"] = name_type
            if len(parts) > 1:
                p["default"] = parts[1].strip()
            params.append(p)
        return params


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------

class CodeIntelligenceTool:
    """Project-wide code intelligence: symbol extraction, call graphs,
    type analysis, and smart navigation."""

    name = "code_intelligence"
    description = (
        "Deep code analysis: extract symbols, build call graphs, "
        "find definitions/references, analyze types and dependencies"
    )

    def __init__(self):
        # Indexed data
        self._symbols: List[SymbolInfo] = []
        self._imports: List[ImportInfo] = []
        self._indexed_root: Optional[str] = None

        # Lookup indices (built after indexing)
        self._symbols_by_name: Dict[str, List[SymbolInfo]] = defaultdict(list)
        self._symbols_by_file: Dict[str, List[SymbolInfo]] = defaultdict(list)
        self._imports_by_file: Dict[str, List[ImportInfo]] = defaultdict(list)
        self._callees_by_func: Dict[str, List[str]] = defaultdict(list)  # func -> [called names]

        # Extractors
        self._py_extractor = _PythonExtractor()
        self._js_extractor = _JSTSExtractor()

    # ==================================================================
    # Indexing
    # ==================================================================

    def index_project(self, root_path: str, languages: List[str] = None) -> dict:
        """Parse all source files, extract symbols, imports, and build indices.

        Args:
            root_path: Project root directory
            languages: Languages to index (default: python, javascript, typescript)

        Returns:
            {success, files_indexed, symbols_found, imports_found, time_ms}
        """
        start = time.time()

        if languages is None:
            languages = ["python", "javascript", "typescript"]

        root = Path(root_path).resolve()
        if not root.exists():
            return {"success": False, "error": f"Path not found: {root_path}"}

        root_str = str(root)
        for blocked in _BLOCKED_ROOTS:
            if root_str.lower() == blocked.lower():
                return {"success": False, "error": f"Indexing system directory '{root_str}' is not allowed"}

        # Reset
        self._symbols = []
        self._imports = []
        self._indexed_root = root_str
        self._symbols_by_name = defaultdict(list)
        self._symbols_by_file = defaultdict(list)
        self._imports_by_file = defaultdict(list)
        self._callees_by_func = defaultdict(list)

        files_indexed = 0
        errors = 0

        for fpath in self._walk_source_files(root, languages):
            try:
                source = fpath.read_text(encoding="utf-8", errors="ignore")
            except (OSError, PermissionError):
                errors += 1
                continue

            rel = str(fpath.relative_to(root))
            lang = EXT_LANG.get(fpath.suffix.lower())

            try:
                if lang == "python":
                    syms, imps = self._py_extractor.extract(source, rel)
                elif lang in ("javascript", "typescript"):
                    syms, imps = self._js_extractor.extract(source, rel)
                else:
                    continue

                self._symbols.extend(syms)
                self._imports.extend(imps)
                files_indexed += 1
            except Exception as e:
                logger.debug("[CodeIntel] Error parsing %s: %s", fpath, e)
                errors += 1

        # Build indices
        self._build_indices()

        elapsed = int((time.time() - start) * 1000)
        logger.info(
            "[CodeIntel] Indexed %d files, %d symbols, %d imports in %dms",
            files_indexed, len(self._symbols), len(self._imports), elapsed,
        )

        return {
            "success": True,
            "files_indexed": files_indexed,
            "symbols_found": len(self._symbols),
            "imports_found": len(self._imports),
            "time_ms": elapsed,
            "errors": errors,
        }

    def _build_indices(self):
        """Build fast lookup indices from extracted data."""
        self._symbols_by_name.clear()
        self._symbols_by_file.clear()
        self._imports_by_file.clear()
        self._callees_by_func.clear()

        for sym in self._symbols:
            # Index by short name (without class prefix)
            short = sym.name.split(".")[-1] if sym.name else ""
            self._symbols_by_name[short].append(sym)
            self._symbols_by_name[sym.name].append(sym)
            self._symbols_by_file[sym.file].append(sym)

            # Build callees index
            if sym.body_names_called:
                self._callees_by_func[sym.name] = sym.body_names_called

        for imp in self._imports:
            self._imports_by_file[imp.file].append(imp)

    def _walk_source_files(self, root: Path, languages: List[str]):
        """Yield source files matching the requested languages."""
        target_exts = {ext for ext, lang in EXT_LANG.items() if lang in languages}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for fname in filenames:
                fpath = Path(dirpath) / fname
                if fpath.suffix.lower() in target_exts:
                    try:
                        if fpath.stat().st_size <= MAX_FILE_SIZE:
                            yield fpath
                    except OSError:
                        pass

    # ==================================================================
    # 1. Symbol extraction queries
    # ==================================================================

    def get_symbols(self, file: str = None, kind: str = None,
                    name_pattern: str = None, limit: int = 200) -> List[dict]:
        """Query extracted symbols with optional filters.

        Args:
            file: Filter by relative file path (substring match)
            kind: Filter by kind (function, method, class, variable, constant,
                  interface, type, enum)
            name_pattern: Regex pattern to match symbol names
            limit: Max results

        Returns:
            List of symbol dicts
        """
        results = self._symbols
        if file:
            results = [s for s in results if file in s.file]
        if kind:
            results = [s for s in results if s.kind == kind]
        if name_pattern:
            try:
                pat = re.compile(name_pattern, re.IGNORECASE)
                results = [s for s in results if pat.search(s.name or "")]
            except re.error:
                pass
        return [s.to_dict() for s in results[:limit]]

    def get_imports(self, file: str = None, module_pattern: str = None,
                    limit: int = 200) -> List[dict]:
        """Query extracted imports.

        Args:
            file: Filter by file path (substring match)
            module_pattern: Regex to match imported module names
            limit: Max results

        Returns:
            List of import dicts
        """
        results = self._imports
        if file:
            results = [i for i in results if file in i.file]
        if module_pattern:
            try:
                pat = re.compile(module_pattern, re.IGNORECASE)
                results = [i for i in results if pat.search(i.module or "")]
            except re.error:
                pass
        return [i.to_dict() for i in results[:limit]]

    # ==================================================================
    # 2. Call graph
    # ==================================================================

    def get_callees(self, function_name: str) -> dict:
        """Get what a function calls (its callees).

        Args:
            function_name: Function name (e.g. 'process_data' or 'MyClass.method')

        Returns:
            {function, callees: [names], resolved: [{name, file, line, kind}]}
        """
        # Direct lookup
        callees = self._callees_by_func.get(function_name, [])

        # Also try class.method format
        if not callees:
            for key, val in self._callees_by_func.items():
                if key.endswith(f".{function_name}") or key == function_name:
                    callees = val
                    break

        # Resolve callees to actual symbols
        resolved = []
        for name in callees:
            matches = self._symbols_by_name.get(name, [])
            for sym in matches:
                resolved.append({
                    "name": sym.name, "file": sym.file,
                    "line": sym.line, "kind": sym.kind,
                })

        return {
            "function": function_name,
            "callees": callees,
            "resolved": resolved,
        }

    def get_callers(self, function_name: str) -> dict:
        """Get what calls a function (its callers).

        Scans all indexed function bodies for references to the target.

        Args:
            function_name: Name to search for in call sites

        Returns:
            {function, callers: [{name, file, line, kind}]}
        """
        short_name = function_name.split(".")[-1]
        callers = []

        for func_name, callees in self._callees_by_func.items():
            if short_name in callees or function_name in callees:
                matches = self._symbols_by_name.get(func_name, [])
                for sym in matches:
                    callers.append({
                        "name": sym.name, "file": sym.file,
                        "line": sym.line, "kind": sym.kind,
                    })

        # Deduplicate
        seen = set()
        unique = []
        for c in callers:
            key = (c["file"], c["line"])
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return {"function": function_name, "callers": unique}

    def get_call_chain(self, function_name: str, max_depth: int = 5) -> dict:
        """Get the dependency chain: A calls B calls C...

        Args:
            function_name: Starting function
            max_depth: Maximum chain depth

        Returns:
            {function, chain: {name: [callees...], ...}, depth}
        """
        chain: Dict[str, List[str]] = {}
        visited: Set[str] = set()
        queue = [function_name]
        depth = 0

        while queue and depth < max_depth:
            next_queue = []
            for name in queue:
                if name in visited:
                    continue
                visited.add(name)
                callees = self._callees_by_func.get(name, [])
                if not callees:
                    # Try short name match
                    for key, val in self._callees_by_func.items():
                        if key.endswith(f".{name}") or key == name:
                            callees = val
                            break
                if callees:
                    chain[name] = callees
                    next_queue.extend(callees)
            queue = next_queue
            depth += 1

        return {"function": function_name, "chain": chain, "depth": depth}

    # ==================================================================
    # 3. Type analysis (Python)
    # ==================================================================

    def get_type_info(self, symbol_name: str) -> dict:
        """Extract type annotation info for a symbol.

        Returns param types, return type, and class attribute types.

        Args:
            symbol_name: Symbol to look up

        Returns:
            {symbol, definitions: [{file, line, params, return_type, kind}]}
        """
        matches = self._symbols_by_name.get(symbol_name, [])
        results = []
        for sym in matches:
            entry: Dict[str, Any] = {
                "file": sym.file,
                "line": sym.line,
                "kind": sym.kind,
                "name": sym.name,
            }
            if sym.params:
                typed_params = [p for p in sym.params if "type" in p]
                if typed_params:
                    entry["params"] = typed_params
            if sym.return_type:
                entry["return_type"] = sym.return_type
            results.append(entry)
        return {"symbol": symbol_name, "definitions": results}

    # ==================================================================
    # 4. Smart navigation
    # ==================================================================

    def find_definition(self, symbol_name: str) -> dict:
        """Find where a symbol is defined.

        First checks the in-memory index (fast). If the project hasn't been
        indexed, falls back to a grep-based scan of the indexed root.

        Args:
            symbol_name: Name to find (function, class, variable, etc.)

        Returns:
            {symbol, definitions: [{file, line, kind, sig, ...}]}
        """
        # Try index first
        matches = self._symbols_by_name.get(symbol_name, [])
        if matches:
            return {
                "symbol": symbol_name,
                "definitions": [s.to_dict() for s in matches],
            }

        # Fallback: grep the project root
        if self._indexed_root:
            return self._grep_definition(symbol_name)

        return {"symbol": symbol_name, "definitions": []}

    def _grep_definition(self, symbol_name: str) -> dict:
        """Fallback definition finder using regex across files."""
        escaped = re.escape(symbol_name)
        patterns = [
            (re.compile(r'^\s*(async\s+)?def\s+' + escaped + r'\s*\(', re.MULTILINE), "function"),
            (re.compile(r'^\s*class\s+' + escaped + r'[\s(:]', re.MULTILINE), "class"),
            (re.compile(r'^' + escaped + r'\s*=', re.MULTILINE), "variable"),
            (re.compile(r'^\s*(export\s+)?(default\s+)?(async\s+)?function\s+' + escaped + r'\s*[\(<]', re.MULTILINE), "function"),
            (re.compile(r'^\s*(export\s+)?(default\s+)?class\s+' + escaped + r'[\s{<]', re.MULTILINE), "class"),
            (re.compile(r'^\s*(export\s+)?(const|let|var)\s+' + escaped + r'\s*[=:]', re.MULTILINE), "variable"),
            (re.compile(r'^\s*(export\s+)?interface\s+' + escaped + r'[\s{<]', re.MULTILINE), "interface"),
            (re.compile(r'^\s*(export\s+)?type\s+' + escaped + r'\s*[=<]', re.MULTILINE), "type"),
        ]

        root = Path(self._indexed_root)
        definitions = []
        for fpath in self._walk_source_files(root, ["python", "javascript", "typescript"]):
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except (OSError, PermissionError):
                continue
            rel = str(fpath.relative_to(root))
            for pat, kind in patterns:
                for m in pat.finditer(text):
                    line = text[:m.start()].count("\n") + 1
                    definitions.append({
                        "file": rel, "line": line,
                        "kind": kind, "sig": m.group(0).strip()[:200],
                    })
        return {"symbol": symbol_name, "definitions": definitions}

    def find_references(self, symbol_name: str, include_definition: bool = False) -> dict:
        """Find all usages of a symbol across the indexed project.

        Args:
            symbol_name: Symbol name to search for
            include_definition: Include the definition site in results

        Returns:
            {symbol, references: [{file, line, text, is_definition}]}
        """
        if not self._indexed_root:
            return {"symbol": symbol_name, "references": []}

        root = Path(self._indexed_root)
        pattern = re.compile(r'\b' + re.escape(symbol_name) + r'\b')

        # Get definition locations for filtering
        def_locations = set()
        if not include_definition:
            for sym in self._symbols_by_name.get(symbol_name, []):
                def_locations.add((sym.file, sym.line))

        references = []
        for fpath in self._walk_source_files(root, ["python", "javascript", "typescript"]):
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except (OSError, PermissionError):
                continue

            rel = str(fpath.relative_to(root))
            for i, line in enumerate(text.split("\n"), 1):
                if pattern.search(line):
                    is_def = (rel, i) in def_locations
                    if is_def and not include_definition:
                        continue
                    references.append({
                        "file": rel, "line": i,
                        "text": line.strip()[:200],
                        "is_definition": is_def,
                    })

        return {
            "symbol": symbol_name,
            "references": references,
            "total": len(references),
        }

    def get_context(self, file: str, line: int) -> dict:
        """Return the enclosing function/class for a given file:line.

        Args:
            file: Relative file path
            line: Line number

        Returns:
            {file, line, enclosing_function, enclosing_class, symbols_at_line}
        """
        file_syms = self._symbols_by_file.get(file, [])
        if not file_syms:
            # Try substring match
            for key, syms in self._symbols_by_file.items():
                if file in key or key.endswith(file):
                    file_syms = syms
                    file = key
                    break

        enclosing_function = None
        enclosing_class = None
        at_line = []

        for sym in file_syms:
            end = sym.end_line or (sym.line + 1)
            if sym.line <= line <= end:
                if sym.kind in ("function", "method"):
                    enclosing_function = sym.to_dict()
                elif sym.kind == "class":
                    enclosing_class = sym.to_dict()
            if sym.line == line:
                at_line.append(sym.to_dict())

        return {
            "file": file,
            "line": line,
            "enclosing_function": enclosing_function,
            "enclosing_class": enclosing_class,
            "symbols_at_line": at_line,
        }

    def get_dependencies(self, file: str) -> dict:
        """Return what a file imports and what other files import from it.

        Args:
            file: Relative file path

        Returns:
            {file, imports_from: [{module, names}], imported_by: [{file, names}]}
        """
        # Normalize: try exact match first, then substring
        file_imports = self._imports_by_file.get(file, [])
        if not file_imports:
            for key, imps in self._imports_by_file.items():
                if file in key or key.endswith(file):
                    file_imports = imps
                    file = key
                    break

        # What this file imports
        imports_from = []
        for imp in file_imports:
            imports_from.append({
                "module": imp.module,
                "names": imp.names,
                "line": imp.line,
                "is_relative": (imp.level or 0) > 0,
            })

        # What imports this file (reverse lookup)
        # Derive possible module names from the file path
        file_stem = Path(file).stem
        file_module = file.replace("/", ".").replace("\\", ".").rsplit(".", 1)[0]
        possible_names = {file_stem, file_module}
        # Also add parent.stem format (e.g., "tools.code_intelligence")
        parts = Path(file).parts
        for i in range(len(parts)):
            mod = ".".join(p.replace(".py", "").replace(".js", "").replace(".ts", "")
                          for p in parts[i:])
            possible_names.add(mod)

        imported_by = []
        for other_file, imps in self._imports_by_file.items():
            if other_file == file:
                continue
            for imp in imps:
                mod = imp.module or ""
                if mod in possible_names or file_stem in mod:
                    imported_by.append({
                        "file": other_file,
                        "module": mod,
                        "names": imp.names,
                        "line": imp.line,
                    })

        return {
            "file": file,
            "imports_from": imports_from,
            "imported_by": imported_by,
        }

    # ==================================================================
    # Repo map (preserved from original)
    # ==================================================================

    def get_repo_map(self, active_files: Optional[List[str]] = None,
                     max_tokens: int = 5000) -> str:
        """Generate a compressed repo map for LLM context.

        Format:
          path/file.py:
          | class Foo(Base):
          |   def bar(self, x: int) -> str
          |   def baz(self)

        Args:
            active_files: Files to prioritize (shown first)
            max_tokens: Approximate token budget (4 chars per token)

        Returns:
            Formatted repo map string
        """
        if not self._symbols:
            return ""

        max_chars = max(1, max_tokens) * 4
        active_set = set(active_files or [])

        # Group by file
        by_file: Dict[str, List[SymbolInfo]] = defaultdict(list)
        for sym in self._symbols:
            by_file[sym.file].append(sym)

        def score(fname):
            return (0 if fname in active_set else 1, fname)

        lines = ["# Repo Map\n"]
        chars = len(lines[0])

        for fname in sorted(by_file.keys(), key=score):
            file_lines = [f"\n{fname}:\n"]
            for sym in sorted(by_file[fname], key=lambda s: s.line or 0):
                if sym.kind == "class":
                    file_lines.append(f"| {sym.sig}\n")
                    # Show methods indented under class
                    if sym.methods:
                        for method in sym.methods:
                            file_lines.append(f"|   {method}\n")
                elif sym.kind in ("function", "method"):
                    # Skip methods already shown under class
                    if "." in (sym.name or ""):
                        continue
                    file_lines.append(f"|   {sym.sig}\n")
                elif sym.kind in ("constant", "variable"):
                    file_lines.append(f"| {sym.sig}\n")
                elif sym.kind in ("interface", "type", "enum"):
                    file_lines.append(f"| {sym.sig}\n")

            chunk = "".join(file_lines)
            if chars + len(chunk) > max_chars:
                break
            lines.append(chunk)
            chars += len(chunk)

        return "".join(lines)

    # ==================================================================
    # Semantic search (preserved from original)
    # ==================================================================

    def semantic_search(self, query: str, limit: int = 10) -> List[dict]:
        """Text-based symbol search. Scores by name/sig/file match quality.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of symbol dicts with scores
        """
        if not self._symbols:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored = []

        for sym in self._symbols:
            score = 0
            name_lower = (sym.name or "").lower()
            sig_lower = (sym.sig or "").lower()
            file_lower = (sym.file or "").lower()
            doc_lower = (sym.docstring or "").lower()

            # Exact name match
            if query_lower == name_lower:
                score += 10
            elif query_lower in name_lower:
                score += 5
            # Word overlap in name
            name_words = set(re.split(r'[_.\s]+', name_lower))
            score += len(query_words & name_words) * 2
            # Sig match
            if query_lower in sig_lower:
                score += 3
            # Docstring match
            if query_lower in doc_lower:
                score += 2
            # File match
            if query_lower in file_lower:
                score += 1

            if score > 0:
                d = sym.to_dict()
                d["_score"] = score
                scored.append(d)

        scored.sort(key=lambda x: -x["_score"])
        return scored[:limit]

    # ==================================================================
    # Legacy compatibility
    # ==================================================================

    def find_callers(self, function_name: str) -> List[str]:
        """Legacy: find files referencing a function name.

        For richer results, use get_callers() instead.
        """
        result = self.get_callers(function_name)
        return list({c["file"] for c in result["callers"]})

    # ==================================================================
    # Summary / stats
    # ==================================================================

    def stats(self) -> dict:
        """Return index statistics."""
        kind_counts: Dict[str, int] = defaultdict(int)
        for sym in self._symbols:
            kind_counts[sym.kind] += 1

        lang_counts: Dict[str, int] = defaultdict(int)
        for sym in self._symbols:
            ext = Path(sym.file).suffix.lower()
            lang = EXT_LANG.get(ext, "unknown")
            lang_counts[lang] += 1

        return {
            "indexed_root": self._indexed_root,
            "total_symbols": len(self._symbols),
            "total_imports": len(self._imports),
            "files": len(self._symbols_by_file),
            "by_kind": dict(kind_counts),
            "by_language": dict(lang_counts),
            "functions_with_call_info": len(self._callees_by_func),
        }

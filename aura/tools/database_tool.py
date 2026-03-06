"""Database tool — SQLite querying, schema inspection, and CSV import/export."""

import csv
import json
import logging
import os
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

_SAFE_TABLE_NAME = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

def _validate_table_name(name: str) -> tuple:
    """Return (valid, error_msg) tuple. Does not raise."""
    if not _SAFE_TABLE_NAME.match(name):
        return False, f"Invalid table name: {name!r}. Use only letters, numbers, underscores."
    return True, None

# Max rows returned per query
MAX_ROWS = 500
# Max databases tracked
MAX_DBS = 20

DB_DIR = Path(__file__).parent.parent.parent / "data" / "databases"


class DatabaseTool:
    """Query SQLite databases, inspect schemas, import/export CSV."""

    name = "database"
    description = "Query SQLite databases, inspect schemas, import/export CSV"

    ALLOWED_SQL_VERBS = {"SELECT", "PRAGMA", "EXPLAIN"}
    BLOCKED_SQL_VERBS = {"DROP", "DELETE", "UPDATE", "INSERT", "CREATE", "ALTER", "ATTACH", "DETACH", "REPLACE"}
    ALLOWED_PRAGMAS = {"table_info", "index_list", "foreign_key_list", "table_xinfo"}

    def _check_sql_safety(self, sql: str) -> tuple:
        """Return (valid, error_msg) tuple. Does not raise."""
        first_word = sql.strip().split()[0].upper() if sql.strip() else ""
        if first_word in self.BLOCKED_SQL_VERBS:
            return False, f"SQL verb '{first_word}' is not allowed. Only SELECT and PRAGMA are permitted."
        if first_word == "PRAGMA":
            # Extract pragma name: PRAGMA pragma_name or PRAGMA pragma_name(args)
            pragma_rest = sql.strip()[len("PRAGMA"):].strip()
            pragma_name = re.split(r'[\s(=]', pragma_rest, maxsplit=1)[0].lower() if pragma_rest else ""
            if pragma_name not in self.ALLOWED_PRAGMAS:
                return False, (
                    f"PRAGMA '{pragma_name}' is not allowed. "
                    f"Allowed PRAGMAs: {', '.join(sorted(self.ALLOWED_PRAGMAS))}"
                )
        return True, None

    def __init__(self):
        DB_DIR.mkdir(parents=True, exist_ok=True)
        self._connections: Dict[str, str] = {}  # alias -> path

    def _get_db_path(self, db: str) -> tuple:
        """Return (path_str, error_msg) tuple. path_str is None on error."""
        from pathlib import Path
        DB_DIR = Path(__file__).parent.parent.parent / "data" / "databases"
        DB_DIR.mkdir(parents=True, exist_ok=True)
        p = Path(db)
        try:
            resolved = (DB_DIR / p.name).resolve()
            if not (str(resolved).startswith(str(DB_DIR.resolve()) + os.sep) or str(resolved) == str(DB_DIR.resolve())):
                return None, f"Database path outside sandbox: {db}"
        except Exception:
            return None, f"Invalid database path: {db}"
        if resolved.suffix not in (".db", ".sqlite", ".sqlite3"):
            resolved = resolved.with_suffix(".db")
        return str(resolved), None

    def _execute_query(self, db_path: str, query: str, params: tuple = ()) -> dict:
        """Execute a SQL query and return results."""
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)

            is_select = query.strip().upper().startswith(("SELECT", "PRAGMA", "EXPLAIN"))

            if is_select:
                rows = cursor.fetchmany(MAX_ROWS + 1)
                truncated = len(rows) > MAX_ROWS
                if truncated:
                    rows = rows[:MAX_ROWS]

                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                data = [dict(row) for row in rows]

                conn.close()
                return {
                    "success": True,
                    "columns": columns,
                    "rows": data,
                    "row_count": len(data),
                    "truncated": truncated,
                }
            else:
                conn.commit()
                affected = cursor.rowcount
                conn.close()
                return {
                    "success": True,
                    "affected_rows": affected,
                    "message": f"{affected} row(s) affected"
                }

        except sqlite3.Error as e:
            return {"success": False, "error": f"SQL error: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Database error: {e}"}

    def query(self, sql: str, db: str = "default") -> dict:
        """Execute a SQL query."""
        if not sql or not sql.strip():
            return {"success": False, "error": "No SQL query provided"}

        valid, err = self._check_sql_safety(sql)
        if not valid:
            return {"success": False, "error": err, "blocked_by": "validation"}

        db_path, err = self._get_db_path(db)
        if err:
            return {"success": False, "error": err, "blocked_by": "validation"}
        result = self._execute_query(db_path, sql)

        if result.get("success") and result.get("rows") is not None:
            # Format as table
            rows = result["rows"]
            cols = result["columns"]
            if rows and cols:
                formatted = self._format_table(cols, rows[:20])
                result["formatted"] = formatted
                result["response"] = f"Query returned {result['row_count']} row(s)" + \
                                     (f" (truncated to {MAX_ROWS})" if result.get("truncated") else "") + \
                                     f"\n{formatted}"
            else:
                result["response"] = "Query returned 0 rows"
        elif result.get("success"):
            result["response"] = result.get("message", "Query executed")
        else:
            result["response"] = result.get("error", "Query failed")

        return result

    def _format_table(self, columns: List[str], rows: List[dict], max_col_width: int = 30) -> str:
        """Format rows as an ASCII table."""
        if not rows:
            return "(empty)"

        # Calculate column widths
        widths = {col: min(max(len(col), max(len(str(row.get(col, ""))[:max_col_width]) for row in rows)), max_col_width)
                  for col in columns}

        # Header
        header = " | ".join(col.ljust(widths[col]) for col in columns)
        separator = "-+-".join("-" * widths[col] for col in columns)
        lines = [header, separator]

        for row in rows:
            line = " | ".join(str(row.get(col, ""))[:max_col_width].ljust(widths[col]) for col in columns)
            lines.append(line)

        return "\n".join(lines)

    def schema(self, db: str = "default", table: str = None) -> dict:
        """Inspect database schema."""
        db_path, err = self._get_db_path(db)
        if err:
            return {"success": False, "error": err, "blocked_by": "validation"}

        if table:
            valid, err = _validate_table_name(table)
            if not valid:
                return {"success": False, "error": err, "blocked_by": "validation"}
            result = self._execute_query(db_path, f"PRAGMA table_info({table})")
            if result.get("success"):
                cols = result.get("rows", [])
                formatted = "\n".join(
                    f"  {c['name']} {c['type']}" +
                    (" PRIMARY KEY" if c.get('pk') else "") +
                    (" NOT NULL" if c.get('notnull') else "")
                    for c in cols
                )
                result["response"] = f"Table '{table}' schema:\n{formatted}"
            return result
        else:
            result = self._execute_query(db_path,
                "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name")
            if result.get("success"):
                tables = result.get("rows", [])
                formatted = "\n".join(f"  [{t['type']}] {t['name']}" for t in tables)
                result["response"] = f"Database '{db}' ({len(tables)} table(s)):\n{formatted}" if tables else "Database is empty"
            return result

    def tables(self, db: str = "default") -> dict:
        """List all tables."""
        return self.schema(db=db)

    def create_db(self, name: str) -> dict:
        """Create a new SQLite database."""
        if not name:
            return {"success": False, "error": "No database name provided"}
        valid, err = _validate_table_name(name)
        if not valid:
            return {"success": False, "error": f"Invalid database name: {err}", "blocked_by": "validation"}
        db_path = str(DB_DIR / (name + ".db"))
        try:
            conn = sqlite3.connect(db_path)
            conn.close()
            self._connections[name] = db_path
            return {"success": True, "path": db_path, "response": f"Database '{name}' created at {db_path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def import_csv(self, csv_path: str, table: str, db: str = "default") -> dict:
        """Import a CSV file into a table."""
        from pathlib import Path
        DATA_DIR = Path(__file__).parent.parent.parent / "data"
        resolved = Path(csv_path).resolve()
        data_dir = DATA_DIR.resolve()
        if not (str(resolved).startswith(str(data_dir) + os.sep) or str(resolved) == str(data_dir)):
            return {"success": False, "error": "CSV path must be within the data directory"}
        p = Path(csv_path)
        if not p.exists():
            return {"success": False, "error": f"File not found: {csv_path}"}

        valid, err = _validate_table_name(table)
        if not valid:
            return {"success": False, "error": err, "blocked_by": "validation"}

        db_path, err = self._get_db_path(db)
        if err:
            return {"success": False, "error": err, "blocked_by": "validation"}

        try:
            with open(p, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                return {"success": False, "error": "CSV file is empty"}

            columns = list(rows[0].keys())
            col_defs = ", ".join(f'"{col}" TEXT' for col in columns)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create table
            cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({col_defs})')

            # Insert rows
            placeholders = ", ".join("?" for _ in columns)
            col_names = ", ".join(f'"{col}"' for col in columns)
            for row in rows:
                values = tuple(row.get(col, "") for col in columns)
                cursor.execute(f'INSERT INTO "{table}" ({col_names}) VALUES ({placeholders})', values)

            conn.commit()
            conn.close()

            return {
                "success": True,
                "table": table,
                "rows_imported": len(rows),
                "columns": columns,
                "response": f"Imported {len(rows)} row(s) into '{table}' ({len(columns)} columns)"
            }

        except Exception as e:
            return {"success": False, "error": f"CSV import failed: {e}"}

    def export_csv(self, table: str, output_path: str = None, db: str = "default") -> dict:
        """Export a table to CSV."""
        valid, err = _validate_table_name(table)
        if not valid:
            return {"success": False, "error": err, "blocked_by": "validation"}

        db_path, err = self._get_db_path(db)
        if err:
            return {"success": False, "error": err, "blocked_by": "validation"}

        result = self._execute_query(db_path, f'SELECT * FROM "{table}"')
        if not result.get("success"):
            return result

        rows = result.get("rows", [])
        columns = result.get("columns", [])

        if not rows:
            return {"success": False, "error": f"Table '{table}' is empty"}

        if not output_path:
            export_dir = DB_DIR / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(export_dir / f"{table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        else:
            resolved_output = Path(output_path).resolve()
            db_dir_resolved = DB_DIR.resolve()
            if not (str(resolved_output).startswith(str(db_dir_resolved) + os.sep) or str(resolved_output) == str(db_dir_resolved)):
                return {"success": False, "error": "Output path must be within the database directory"}

        try:
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)

            return {
                "success": True,
                "path": output_path,
                "rows": len(rows),
                "columns": columns,
                "response": f"Exported {len(rows)} row(s) to {output_path}"
            }

        except Exception as e:
            return {"success": False, "error": f"CSV export failed: {e}"}

    def count(self, table: str, db: str = "default") -> dict:
        """Get row count for a table."""
        valid, err = _validate_table_name(table)
        if not valid:
            return {"success": False, "error": err, "blocked_by": "validation"}
        return self.query(f'SELECT COUNT(*) as count FROM "{table}"', db=db)

    # -- Dispatch -----------------------------------------------------------

    def execute(self, action: str, **kwargs) -> dict:
        action_lower = action.lower().strip()
        db = kwargs.get("db", "default")

        # Schema / tables
        if action_lower.startswith("schema") or action_lower.startswith("tables") or action_lower.startswith("describe"):
            table = kwargs.get("table")
            if not table and len(action.split()) > 1:
                table = action.split(None, 1)[-1].strip()
            return self.schema(db=db, table=table if table and table != "tables" else None)

        # Create database
        if action_lower.startswith("create_db") or action_lower.startswith("create database"):
            name = kwargs.get("name") or (action.split(None, 2)[-1] if len(action.split()) > 2 else None)
            if name:
                return self.create_db(name)
            return {"success": False, "error": "No database name specified"}

        # Import CSV
        if action_lower.startswith("import"):
            csv_path = kwargs.get("csv_path") or kwargs.get("path")
            table = kwargs.get("table")
            if not csv_path or not table:
                # Try: "import <path> <table>"
                parts = action.split()
                if len(parts) >= 3:
                    csv_path = csv_path or parts[1]
                    table = table or parts[2]
            if csv_path and table:
                return self.import_csv(csv_path, table, db=db)
            return {"success": False, "error": "Usage: import <csv_path> <table_name>"}

        # Export CSV
        if action_lower.startswith("export"):
            table = kwargs.get("table")
            output_path = kwargs.get("output_path") or kwargs.get("path")
            if not table and len(action.split()) > 1:
                table = action.split(None, 1)[-1].strip()
            if table:
                return self.export_csv(table, output_path=output_path, db=db)
            return {"success": False, "error": "Usage: export <table_name>"}

        # Count
        if action_lower.startswith("count"):
            table = kwargs.get("table")
            if not table and len(action.split()) > 1:
                table = action.split(None, 1)[-1].strip()
            if table:
                return self.count(table, db=db)
            return {"success": False, "error": "Usage: count <table_name>"}

        # Default: treat as SQL query
        sql = kwargs.get("sql") or action
        return self.query(sql, db=db)


# Singleton
database_tool = DatabaseTool()

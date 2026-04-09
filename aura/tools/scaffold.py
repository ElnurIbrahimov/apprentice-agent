"""Scaffold Tool — generate complete project templates from built-in stacks or AI-driven custom generation.

Like create-next-app but for any stack, powered by Aura's brain for custom requests.

Built-in templates (instant, no AI):
  nextjs, react-vite, fastapi, express, static, extension

AI-powered:
  custom — describe what you want and Aura picks the stack + generates all files.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tool_contract import ToolResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, Dict[str, Any]] = {
    "nextjs": {
        "label": "Next.js + Tailwind + TypeScript",
        "description": "Full-stack React framework with Tailwind CSS and TypeScript",
        "tags": ["frontend", "fullstack", "react", "typescript"],
    },
    "react-vite": {
        "label": "React + Vite + TypeScript",
        "description": "Fast React SPA with Vite bundler, TypeScript, and Tailwind CSS",
        "tags": ["frontend", "spa", "react", "typescript"],
    },
    "fastapi": {
        "label": "FastAPI + Python",
        "description": "Modern async Python API with structured project layout",
        "tags": ["backend", "api", "python"],
    },
    "express": {
        "label": "Express + Node.js + TypeScript",
        "description": "Node.js REST API with Express, TypeScript, and clean project structure",
        "tags": ["backend", "api", "node", "typescript"],
    },
    "static": {
        "label": "Static HTML + CSS + JS",
        "description": "Clean, modern static site with responsive layout",
        "tags": ["frontend", "static", "html"],
    },
    "extension": {
        "label": "Chrome Extension (Manifest V3)",
        "description": "Browser extension with popup, background script, and content script",
        "tags": ["extension", "chrome", "browser"],
    },
    "custom": {
        "label": "Custom (AI-generated)",
        "description": "Describe what you want — Aura picks the best stack and generates everything",
        "tags": ["ai", "custom"],
    },
}


# ---------------------------------------------------------------------------
# File generators per template
# ---------------------------------------------------------------------------

def _gen_nextjs(name: str, opts: Dict[str, Any]) -> Dict[str, str]:
    """Generate Next.js + Tailwind + TypeScript project files."""
    safe_name = _slugify(name)
    desc = opts.get("description", "A Next.js application")

    files: Dict[str, str] = {}

    files["package.json"] = json.dumps({
        "name": safe_name,
        "version": "0.1.0",
        "private": True,
        "scripts": {
            "dev": "next dev",
            "build": "next build",
            "start": "next start",
            "lint": "next lint"
        },
        "dependencies": {
            "next": "^15.1.0",
            "react": "^19.0.0",
            "react-dom": "^19.0.0"
        },
        "devDependencies": {
            "@types/node": "^22.0.0",
            "@types/react": "^19.0.0",
            "@types/react-dom": "^19.0.0",
            "typescript": "^5.7.0",
            "tailwindcss": "^4.0.0",
            "@tailwindcss/postcss": "^4.0.0",
            "postcss": "^8.5.0",
            "eslint": "^9.0.0",
            "eslint-config-next": "^15.1.0"
        }
    }, indent=2)

    files["tsconfig.json"] = json.dumps({
        "compilerOptions": {
            "target": "ES2017",
            "lib": ["dom", "dom.iterable", "esnext"],
            "allowJs": True,
            "skipLibCheck": True,
            "strict": True,
            "noEmit": True,
            "esModuleInterop": True,
            "module": "esnext",
            "moduleResolution": "bundler",
            "resolveJsonModule": True,
            "isolatedModules": True,
            "jsx": "preserve",
            "incremental": True,
            "plugins": [{"name": "next"}],
            "paths": {"@/*": ["./*"]}
        },
        "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
        "exclude": ["node_modules"]
    }, indent=2)

    files["next.config.ts"] = textwrap.dedent("""\
        import type { NextConfig } from "next";

        const nextConfig: NextConfig = {
          /* config options here */
        };

        export default nextConfig;
    """)

    files["postcss.config.mjs"] = textwrap.dedent("""\
        const config = {
          plugins: {
            "@tailwindcss/postcss": {},
          },
        };

        export default config;
    """)

    files["app/globals.css"] = textwrap.dedent("""\
        @import "tailwindcss";

        :root {
          --background: #ffffff;
          --foreground: #171717;
        }

        @media (prefers-color-scheme: dark) {
          :root {
            --background: #0a0a0a;
            --foreground: #ededed;
          }
        }

        body {
          color: var(--foreground);
          background: var(--background);
          font-family: system-ui, -apple-system, sans-serif;
        }
    """)

    files["app/layout.tsx"] = textwrap.dedent(f"""\
        import type {{ Metadata }} from "next";
        import "./globals.css";

        export const metadata: Metadata = {{
          title: "{name}",
          description: "{desc}",
        }};

        export default function RootLayout({{
          children,
        }}: Readonly<{{
          children: React.ReactNode;
        }}>) {{
          return (
            <html lang="en">
              <body className="antialiased">
                {{children}}
              </body>
            </html>
          );
        }}
    """)

    files["app/page.tsx"] = textwrap.dedent(f"""\
        export default function Home() {{
          return (
            <main className="flex min-h-screen flex-col items-center justify-center p-24">
              <h1 className="text-4xl font-bold mb-4">{name}</h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 mb-8">
                {desc}
              </p>
              <div className="flex gap-4">
                <a
                  href="https://nextjs.org/docs"
                  className="rounded-lg bg-black text-white dark:bg-white dark:text-black px-6 py-3 font-medium hover:opacity-80 transition"
                >
                  Docs
                </a>
                <a
                  href="https://nextjs.org/learn"
                  className="rounded-lg border border-gray-300 dark:border-gray-700 px-6 py-3 font-medium hover:bg-gray-50 dark:hover:bg-gray-900 transition"
                >
                  Learn
                </a>
              </div>
            </main>
          );
        }}
    """)

    files["public/.gitkeep"] = ""

    files[".gitignore"] = textwrap.dedent("""\
        # dependencies
        /node_modules
        /.pnp
        .pnp.*

        # next.js
        /.next/
        /out/

        # production
        /build

        # misc
        .DS_Store
        *.pem

        # debug
        npm-debug.log*
        yarn-debug.log*
        yarn-error.log*
        .pnpm-debug.log*

        # env
        .env*.local

        # typescript
        *.tsbuildinfo
        next-env.d.ts
    """)

    files["README.md"] = textwrap.dedent(f"""\
        # {name}

        {desc}

        ## Getting Started

        ```bash
        npm install
        npm run dev
        ```

        Open [http://localhost:3000](http://localhost:3000) in your browser.
    """)

    return files


def _gen_react_vite(name: str, opts: Dict[str, Any]) -> Dict[str, str]:
    """Generate React + Vite + TypeScript project files."""
    safe_name = _slugify(name)
    desc = opts.get("description", "A React application")

    files: Dict[str, str] = {}

    files["package.json"] = json.dumps({
        "name": safe_name,
        "private": True,
        "version": "0.1.0",
        "type": "module",
        "scripts": {
            "dev": "vite",
            "build": "tsc -b && vite build",
            "lint": "eslint .",
            "preview": "vite preview"
        },
        "dependencies": {
            "react": "^19.0.0",
            "react-dom": "^19.0.0"
        },
        "devDependencies": {
            "@types/react": "^19.0.0",
            "@types/react-dom": "^19.0.0",
            "@vitejs/plugin-react": "^4.3.0",
            "typescript": "~5.7.0",
            "vite": "^6.0.0",
            "tailwindcss": "^4.0.0",
            "@tailwindcss/vite": "^4.0.0"
        }
    }, indent=2)

    files["tsconfig.json"] = json.dumps({
        "compilerOptions": {
            "target": "ES2020",
            "useDefineForClassFields": True,
            "lib": ["ES2020", "DOM", "DOM.Iterable"],
            "module": "ESNext",
            "skipLibCheck": True,
            "moduleResolution": "bundler",
            "allowImportingTsExtensions": True,
            "isolatedModules": True,
            "moduleDetection": "force",
            "noEmit": True,
            "jsx": "react-jsx",
            "strict": True,
            "noUnusedLocals": True,
            "noUnusedParameters": True,
            "noFallthroughCasesInSwitch": True,
            "noUncheckedSideEffectImports": True
        },
        "include": ["src"]
    }, indent=2)

    files["vite.config.ts"] = textwrap.dedent("""\
        import { defineConfig } from "vite";
        import react from "@vitejs/plugin-react";
        import tailwindcss from "@tailwindcss/vite";

        export default defineConfig({
          plugins: [react(), tailwindcss()],
        });
    """)

    files["index.html"] = textwrap.dedent(f"""\
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>{name}</title>
          </head>
          <body>
            <div id="root"></div>
            <script type="module" src="/src/main.tsx"></script>
          </body>
        </html>
    """)

    files["src/main.tsx"] = textwrap.dedent("""\
        import { StrictMode } from "react";
        import { createRoot } from "react-dom/client";
        import App from "./App";
        import "./index.css";

        createRoot(document.getElementById("root")!).render(
          <StrictMode>
            <App />
          </StrictMode>
        );
    """)

    files["src/index.css"] = textwrap.dedent("""\
        @import "tailwindcss";

        :root {
          --background: #ffffff;
          --foreground: #1a1a1a;
        }

        @media (prefers-color-scheme: dark) {
          :root {
            --background: #0a0a0a;
            --foreground: #f5f5f5;
          }
        }

        body {
          margin: 0;
          color: var(--foreground);
          background: var(--background);
          font-family: system-ui, -apple-system, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
    """)

    files["src/App.tsx"] = textwrap.dedent(f"""\
        import {{ useState }} from "react";

        function App() {{
          const [count, setCount] = useState(0);

          return (
            <main className="flex min-h-screen flex-col items-center justify-center p-8">
              <h1 className="text-4xl font-bold mb-4">{name}</h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 mb-8">
                {desc}
              </p>
              <div className="flex flex-col items-center gap-4">
                <button
                  onClick={{() => setCount((c) => c + 1)}}
                  className="rounded-lg bg-indigo-600 text-white px-6 py-3 font-medium hover:bg-indigo-700 transition"
                >
                  Count: {{count}}
                </button>
                <p className="text-sm text-gray-500">
                  Edit <code className="font-mono bg-gray-100 dark:bg-gray-800 px-1 rounded">src/App.tsx</code> and save to see HMR
                </p>
              </div>
            </main>
          );
        }}

        export default App;
    """)

    files[".gitignore"] = textwrap.dedent("""\
        node_modules
        dist
        dist-ssr
        *.local
        .DS_Store
        .env
        .env.local
    """)

    files["README.md"] = textwrap.dedent(f"""\
        # {name}

        {desc}

        ## Getting Started

        ```bash
        npm install
        npm run dev
        ```

        Open [http://localhost:5173](http://localhost:5173) in your browser.
    """)

    return files


def _gen_fastapi(name: str, opts: Dict[str, Any]) -> Dict[str, str]:
    """Generate FastAPI + Python project files."""
    safe_name = _slugify(name)
    desc = opts.get("description", "A FastAPI application")

    files: Dict[str, str] = {}

    files["requirements.txt"] = textwrap.dedent("""\
        fastapi>=0.115.0
        uvicorn[standard]>=0.32.0
        pydantic>=2.10.0
        pydantic-settings>=2.6.0
        python-dotenv>=1.0.0
        httpx>=0.28.0
    """)

    files["pyproject.toml"] = textwrap.dedent(f"""\
        [project]
        name = "{safe_name}"
        version = "0.1.0"
        description = "{desc}"
        requires-python = ">=3.11"
        dependencies = [
            "fastapi>=0.115.0",
            "uvicorn[standard]>=0.32.0",
            "pydantic>=2.10.0",
            "pydantic-settings>=2.6.0",
            "python-dotenv>=1.0.0",
            "httpx>=0.28.0",
        ]

        [project.scripts]
        dev = "uvicorn app.main:app --reload"
    """)

    files["main.py"] = textwrap.dedent("""\
        \"\"\"Entry point — run with: uvicorn main:app --reload\"\"\"
        from app.main import app  # noqa: F401
    """)

    files["app/__init__.py"] = ""

    files["app/main.py"] = textwrap.dedent(f"""\
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        from app.config import settings
        from app.routes import health, items

        app = FastAPI(
            title="{name}",
            description="{desc}",
            version="0.1.0",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.include_router(health.router)
        app.include_router(items.router, prefix="/api/v1")
    """)

    files["app/config.py"] = textwrap.dedent(f"""\
        from pydantic_settings import BaseSettings


        class Settings(BaseSettings):
            app_name: str = "{name}"
            debug: bool = False
            cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
            database_url: str = "sqlite:///./data.db"

            model_config = {{"env_file": ".env", "env_file_encoding": "utf-8"}}


        settings = Settings()
    """)

    files["app/routes/__init__.py"] = ""

    files["app/routes/health.py"] = textwrap.dedent("""\
        from fastapi import APIRouter

        router = APIRouter(tags=["health"])


        @router.get("/health")
        async def health_check():
            return {"status": "ok"}
    """)

    files["app/routes/items.py"] = textwrap.dedent("""\
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel

        router = APIRouter(tags=["items"])


        class Item(BaseModel):
            id: int | None = None
            name: str
            description: str = ""


        # In-memory store for demo purposes
        _items: dict[int, Item] = {}
        _next_id: int = 1


        @router.get("/items")
        async def list_items():
            return list(_items.values())


        @router.get("/items/{item_id}")
        async def get_item(item_id: int):
            if item_id not in _items:
                raise HTTPException(status_code=404, detail="Item not found")
            return _items[item_id]


        @router.post("/items", status_code=201)
        async def create_item(item: Item):
            global _next_id
            item.id = _next_id
            _items[_next_id] = item
            _next_id += 1
            return item


        @router.delete("/items/{item_id}", status_code=204)
        async def delete_item(item_id: int):
            if item_id not in _items:
                raise HTTPException(status_code=404, detail="Item not found")
            del _items[item_id]
    """)

    files["app/models/__init__.py"] = ""

    files["app/models/base.py"] = textwrap.dedent("""\
        \"\"\"Shared model utilities — add your SQLAlchemy/SQLModel base here when needed.\"\"\"
    """)

    files[".env.example"] = textwrap.dedent(f"""\
        APP_NAME={name}
        DEBUG=true
        DATABASE_URL=sqlite:///./data.db
        CORS_ORIGINS=["http://localhost:3000"]
    """)

    files[".gitignore"] = textwrap.dedent("""\
        __pycache__/
        *.py[cod]
        *$py.class
        .env
        .venv/
        venv/
        *.db
        *.sqlite3
        dist/
        *.egg-info/
        .mypy_cache/
        .ruff_cache/
        .DS_Store
    """)

    files["README.md"] = textwrap.dedent(f"""\
        # {name}

        {desc}

        ## Getting Started

        ```bash
        python -m venv .venv
        source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
        pip install -r requirements.txt
        uvicorn main:app --reload
        ```

        API docs at [http://localhost:8000/docs](http://localhost:8000/docs).
    """)

    return files


def _gen_express(name: str, opts: Dict[str, Any]) -> Dict[str, str]:
    """Generate Express + Node.js + TypeScript project files."""
    safe_name = _slugify(name)
    desc = opts.get("description", "An Express API")

    files: Dict[str, str] = {}

    files["package.json"] = json.dumps({
        "name": safe_name,
        "version": "0.1.0",
        "private": True,
        "scripts": {
            "dev": "tsx watch src/index.ts",
            "build": "tsc",
            "start": "node dist/index.js",
            "lint": "eslint src/"
        },
        "dependencies": {
            "express": "^5.0.0",
            "cors": "^2.8.5",
            "dotenv": "^16.4.0",
            "helmet": "^8.0.0",
            "morgan": "^1.10.0"
        },
        "devDependencies": {
            "@types/express": "^5.0.0",
            "@types/cors": "^2.8.17",
            "@types/morgan": "^1.9.9",
            "@types/node": "^22.0.0",
            "typescript": "^5.7.0",
            "tsx": "^4.19.0"
        }
    }, indent=2)

    files["tsconfig.json"] = json.dumps({
        "compilerOptions": {
            "target": "ES2022",
            "module": "NodeNext",
            "moduleResolution": "NodeNext",
            "lib": ["ES2022"],
            "outDir": "./dist",
            "rootDir": "./src",
            "strict": True,
            "esModuleInterop": True,
            "skipLibCheck": True,
            "forceConsistentCasingInFileNames": True,
            "resolveJsonModule": True,
            "declaration": True,
            "declarationMap": True,
            "sourceMap": True
        },
        "include": ["src/**/*"],
        "exclude": ["node_modules", "dist"]
    }, indent=2)

    files["src/index.ts"] = textwrap.dedent(f"""\
        import express from "express";
        import cors from "cors";
        import helmet from "helmet";
        import morgan from "morgan";
        import {{ config }} from "./config.js";
        import {{ healthRouter }} from "./routes/health.js";
        import {{ itemsRouter }} from "./routes/items.js";
        import {{ errorHandler }} from "./middleware/error.js";

        const app = express();

        // Middleware
        app.use(helmet());
        app.use(cors());
        app.use(morgan("dev"));
        app.use(express.json());

        // Routes
        app.use("/health", healthRouter);
        app.use("/api/v1/items", itemsRouter);

        // Error handler (must be last)
        app.use(errorHandler);

        app.listen(config.port, () => {{
          console.log(`{name} running on http://localhost:${{config.port}}`);
        }});
    """)

    files["src/config.ts"] = textwrap.dedent("""\
        import "dotenv/config";

        export const config = {
          port: parseInt(process.env.PORT || "3001", 10),
          nodeEnv: process.env.NODE_ENV || "development",
        };
    """)

    files["src/routes/health.ts"] = textwrap.dedent("""\
        import { Router } from "express";

        export const healthRouter = Router();

        healthRouter.get("/", (_req, res) => {
          res.json({ status: "ok", timestamp: new Date().toISOString() });
        });
    """)

    files["src/routes/items.ts"] = textwrap.dedent("""\
        import { Router } from "express";

        export const itemsRouter = Router();

        interface Item {
          id: number;
          name: string;
          description: string;
        }

        const items: Map<number, Item> = new Map();
        let nextId = 1;

        itemsRouter.get("/", (_req, res) => {
          res.json([...items.values()]);
        });

        itemsRouter.get("/:id", (req, res) => {
          const item = items.get(Number(req.params.id));
          if (!item) {
            res.status(404).json({ error: "Item not found" });
            return;
          }
          res.json(item);
        });

        itemsRouter.post("/", (req, res) => {
          const { name, description = "" } = req.body;
          if (!name) {
            res.status(400).json({ error: "name is required" });
            return;
          }
          const item: Item = { id: nextId++, name, description };
          items.set(item.id, item);
          res.status(201).json(item);
        });

        itemsRouter.delete("/:id", (req, res) => {
          const id = Number(req.params.id);
          if (!items.has(id)) {
            res.status(404).json({ error: "Item not found" });
            return;
          }
          items.delete(id);
          res.status(204).send();
        });
    """)

    files["src/middleware/error.ts"] = textwrap.dedent("""\
        import type { Request, Response, NextFunction } from "express";

        export function errorHandler(
          err: Error,
          _req: Request,
          res: Response,
          _next: NextFunction
        ) {
          console.error(err.stack);
          res.status(500).json({ error: "Internal server error" });
        }
    """)

    files[".env.example"] = textwrap.dedent("""\
        PORT=3001
        NODE_ENV=development
    """)

    files[".gitignore"] = textwrap.dedent("""\
        node_modules
        dist
        .env
        .DS_Store
        *.log
    """)

    files["README.md"] = textwrap.dedent(f"""\
        # {name}

        {desc}

        ## Getting Started

        ```bash
        npm install
        npm run dev
        ```

        Server runs at [http://localhost:3001](http://localhost:3001).
        Health check: `GET /health`
        API: `GET/POST/DELETE /api/v1/items`
    """)

    return files


def _gen_static(name: str, opts: Dict[str, Any]) -> Dict[str, str]:
    """Generate static HTML + CSS + JS site."""
    desc = opts.get("description", "A static website")

    files: Dict[str, str] = {}

    files["index.html"] = textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>{name}</title>
          <link rel="stylesheet" href="styles.css" />
        </head>
        <body>
          <header>
            <nav>
              <div class="logo">{name}</div>
              <ul class="nav-links">
                <li><a href="#home" class="active">Home</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#contact">Contact</a></li>
              </ul>
            </nav>
          </header>

          <main>
            <section id="home" class="hero">
              <h1>{name}</h1>
              <p>{desc}</p>
              <a href="#about" class="btn">Learn More</a>
            </section>

            <section id="about" class="section">
              <h2>About</h2>
              <p>This is a clean, modern, and responsive static site. Edit the HTML, CSS, and JS to make it your own.</p>
            </section>

            <section id="contact" class="section">
              <h2>Contact</h2>
              <p>Get in touch at <a href="mailto:hello@example.com">hello@example.com</a></p>
            </section>
          </main>

          <footer>
            <p>&copy; {datetime.now().year} {name}. All rights reserved.</p>
          </footer>

          <script src="script.js"></script>
        </body>
        </html>
    """)

    files["styles.css"] = textwrap.dedent("""\
        *,
        *::before,
        *::after {
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }

        :root {
          --bg: #ffffff;
          --fg: #1a1a1a;
          --accent: #3b82f6;
          --accent-hover: #2563eb;
          --muted: #6b7280;
          --border: #e5e7eb;
          --radius: 8px;
        }

        @media (prefers-color-scheme: dark) {
          :root {
            --bg: #0f172a;
            --fg: #f1f5f9;
            --accent: #60a5fa;
            --accent-hover: #93bbfd;
            --muted: #94a3b8;
            --border: #334155;
          }
        }

        body {
          font-family: system-ui, -apple-system, sans-serif;
          color: var(--fg);
          background: var(--bg);
          line-height: 1.6;
        }

        /* Nav */
        nav {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1rem 2rem;
          border-bottom: 1px solid var(--border);
        }

        .logo {
          font-size: 1.25rem;
          font-weight: 700;
        }

        .nav-links {
          display: flex;
          list-style: none;
          gap: 1.5rem;
        }

        .nav-links a {
          color: var(--muted);
          text-decoration: none;
          font-weight: 500;
          transition: color 0.2s;
        }

        .nav-links a:hover,
        .nav-links a.active {
          color: var(--accent);
        }

        /* Hero */
        .hero {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          min-height: 70vh;
          padding: 2rem;
        }

        .hero h1 {
          font-size: 3rem;
          margin-bottom: 1rem;
        }

        .hero p {
          font-size: 1.25rem;
          color: var(--muted);
          max-width: 600px;
          margin-bottom: 2rem;
        }

        /* Buttons */
        .btn {
          display: inline-block;
          padding: 0.75rem 2rem;
          background: var(--accent);
          color: white;
          border-radius: var(--radius);
          text-decoration: none;
          font-weight: 600;
          transition: background 0.2s;
        }

        .btn:hover {
          background: var(--accent-hover);
        }

        /* Sections */
        .section {
          max-width: 800px;
          margin: 0 auto;
          padding: 4rem 2rem;
        }

        .section h2 {
          font-size: 2rem;
          margin-bottom: 1rem;
        }

        .section p {
          color: var(--muted);
          font-size: 1.1rem;
        }

        /* Footer */
        footer {
          text-align: center;
          padding: 2rem;
          border-top: 1px solid var(--border);
          color: var(--muted);
          font-size: 0.875rem;
        }

        /* Responsive */
        @media (max-width: 640px) {
          .hero h1 { font-size: 2rem; }
          .nav-links { gap: 1rem; }
          nav { padding: 1rem; }
        }
    """)

    files["script.js"] = textwrap.dedent("""\
        // Smooth scroll for anchor links
        document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
          anchor.addEventListener("click", function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute("href"));
            if (target) {
              target.scrollIntoView({ behavior: "smooth" });
            }
          });
        });

        // Active nav link on scroll
        const sections = document.querySelectorAll("section[id]");
        const navLinks = document.querySelectorAll(".nav-links a");

        window.addEventListener("scroll", () => {
          let current = "";
          sections.forEach((section) => {
            const top = section.offsetTop - 100;
            if (window.scrollY >= top) {
              current = section.getAttribute("id");
            }
          });
          navLinks.forEach((link) => {
            link.classList.remove("active");
            if (link.getAttribute("href") === `#${current}`) {
              link.classList.add("active");
            }
          });
        });
    """)

    files[".gitignore"] = textwrap.dedent("""\
        .DS_Store
        Thumbs.db
        *.log
    """)

    return files


def _gen_extension(name: str, opts: Dict[str, Any]) -> Dict[str, str]:
    """Generate Chrome Extension (Manifest V3) project files."""
    desc = opts.get("description", "A Chrome extension")

    files: Dict[str, str] = {}

    files["manifest.json"] = json.dumps({
        "manifest_version": 3,
        "name": name,
        "version": "1.0.0",
        "description": desc,
        "permissions": ["storage", "activeTab"],
        "action": {
            "default_popup": "popup.html",
            "default_icon": {
                "16": "icons/icon16.png",
                "48": "icons/icon48.png",
                "128": "icons/icon128.png"
            }
        },
        "background": {
            "service_worker": "background.js"
        },
        "content_scripts": [{
            "matches": ["<all_urls>"],
            "js": ["content.js"],
            "run_at": "document_idle"
        }],
        "icons": {
            "16": "icons/icon16.png",
            "48": "icons/icon48.png",
            "128": "icons/icon128.png"
        }
    }, indent=2)

    files["popup.html"] = textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
              width: 320px;
              padding: 16px;
              font-family: system-ui, -apple-system, sans-serif;
              background: #ffffff;
              color: #1a1a1a;
            }}
            h1 {{
              font-size: 18px;
              margin-bottom: 12px;
            }}
            p {{
              font-size: 14px;
              color: #6b7280;
              margin-bottom: 16px;
            }}
            .toggle-row {{
              display: flex;
              justify-content: space-between;
              align-items: center;
              padding: 8px 0;
            }}
            button {{
              padding: 8px 16px;
              background: #3b82f6;
              color: white;
              border: none;
              border-radius: 6px;
              cursor: pointer;
              font-size: 14px;
            }}
            button:hover {{ background: #2563eb; }}
            #status {{
              margin-top: 12px;
              padding: 8px;
              border-radius: 6px;
              background: #f0fdf4;
              color: #166534;
              font-size: 13px;
              display: none;
            }}
          </style>
        </head>
        <body>
          <h1>{name}</h1>
          <p>{desc}</p>
          <div class="toggle-row">
            <span>Enabled</span>
            <button id="toggleBtn">Toggle</button>
          </div>
          <div id="status"></div>
          <script src="popup.js"></script>
        </body>
        </html>
    """)

    files["popup.js"] = textwrap.dedent("""\
        const toggleBtn = document.getElementById("toggleBtn");
        const statusEl = document.getElementById("status");

        // Load saved state
        chrome.storage.local.get(["enabled"], (result) => {
          updateUI(result.enabled !== false);
        });

        toggleBtn.addEventListener("click", () => {
          chrome.storage.local.get(["enabled"], (result) => {
            const newState = result.enabled === false;
            chrome.storage.local.set({ enabled: newState });
            updateUI(newState);
            showStatus(newState ? "Extension enabled" : "Extension disabled");
          });
        });

        function updateUI(enabled) {
          toggleBtn.textContent = enabled ? "Disable" : "Enable";
        }

        function showStatus(message) {
          statusEl.textContent = message;
          statusEl.style.display = "block";
          setTimeout(() => { statusEl.style.display = "none"; }, 2000);
        }
    """)

    files["background.js"] = textwrap.dedent("""\
        // Background service worker — runs independently of any page

        chrome.runtime.onInstalled.addListener(() => {
          console.log("Extension installed");
          chrome.storage.local.set({ enabled: true });
        });

        // Listen for messages from content scripts or popup
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
          if (message.type === "GET_STATUS") {
            chrome.storage.local.get(["enabled"], (result) => {
              sendResponse({ enabled: result.enabled !== false });
            });
            return true; // async response
          }
        });
    """)

    files["content.js"] = textwrap.dedent("""\
        // Content script — runs on every page (per manifest matches)

        (function () {
          // Check if extension is enabled before doing anything
          chrome.runtime.sendMessage({ type: "GET_STATUS" }, (response) => {
            if (chrome.runtime.lastError || !response?.enabled) return;

            console.log("Content script loaded on:", window.location.href);

            // Add your page-level logic here.
            // Example: highlight all links
            // document.querySelectorAll("a").forEach(el => {
            //   el.style.outline = "2px solid #3b82f6";
            // });
          });
        })();
    """)

    files["icons/.gitkeep"] = ""

    files[".gitignore"] = textwrap.dedent("""\
        .DS_Store
        *.log
        *.crx
        *.pem
    """)

    files["README.md"] = textwrap.dedent(f"""\
        # {name}

        {desc}

        ## Load in Chrome

        1. Go to `chrome://extensions`
        2. Enable **Developer mode**
        3. Click **Load unpacked** and select this folder
        4. Add icon images to the `icons/` folder (16x16, 48x48, 128x128 PNG)
    """)

    return files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    """Convert project name to a package-safe slug."""
    import re
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
    return slug or "my-project"


def _write_files(base: Path, file_map: Dict[str, str]) -> List[str]:
    """Write a dict of {relative_path: content} to disk. Returns list of created paths."""
    created = []
    base_resolved = base.resolve()
    for rel_path, content in sorted(file_map.items()):
        if ".." in str(rel_path):
            logger.warning(f"[Scaffold] Skipping path with traversal: {rel_path}")
            continue
        full = (base / rel_path).resolve()
        try:
            full.relative_to(base_resolved)
        except ValueError:
            logger.warning(f"[Scaffold] Skipping path outside target: {rel_path}")
            continue
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        created.append(str(full))
    return created


# Map template keys to generators
_GENERATORS = {
    "nextjs": _gen_nextjs,
    "react-vite": _gen_react_vite,
    "fastapi": _gen_fastapi,
    "express": _gen_express,
    "static": _gen_static,
    "extension": _gen_extension,
}

# AI prompt for custom scaffolding
_CUSTOM_SCAFFOLD_PROMPT = textwrap.dedent("""\
You are a project scaffolding generator. The user wants to create a new project.
Based on their description, you must:

1. Pick the best technology stack.
2. Generate a COMPLETE set of project files with WORKING code (not placeholders).
3. Return your answer as a JSON object with this exact structure:

{
  "stack": "short description of chosen stack",
  "files": {
    "relative/path/to/file.ext": "full file content here",
    ...
  },
  "post_install": "shell command to install dependencies, e.g. npm install",
  "summary": "1-2 sentence description of what was generated"
}

Rules:
- Every file must contain real, working code.
- Include a README.md with setup instructions.
- Include a .gitignore.
- Include dependency files (package.json, requirements.txt, etc.).
- Use modern versions of all libraries.
- Use TypeScript for JS projects.
- Return ONLY the JSON object, no markdown fences, no explanation outside the JSON.

User's project description:
""")


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------

class ScaffoldTool:
    """Generate project scaffolding from built-in templates or AI-driven custom generation."""

    name = "scaffold"
    description = "Generate project scaffolding from templates or AI-driven custom generation"

    def __init__(self, brain=None):
        """
        Args:
            brain: Optional OllamaBrain instance for AI-powered custom scaffolding.
                   If None, custom scaffolding will be unavailable.
        """
        self._brain = brain

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        template: Optional[str] = None,
        description: Optional[str] = None,
        path: str = ".",
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        auto_install: bool = False,
    ) -> ToolResult:
        """Main entry point — dispatches to the right generator.

        Args:
            template: One of "nextjs", "react-vite", "fastapi", "express",
                      "static", "extension", "custom", or None to list templates.
            description: Natural language description (required for "custom",
                         optional for others as project description).
            path: Directory to create the project in. Created if missing.
            name: Project name. Defaults to the directory name.
            options: Extra options dict passed to the generator.
            auto_install: If True, run npm install / pip install after scaffolding.
        """
        if template is None:
            return self.list_templates()

        template = template.lower().strip()

        if template == "custom":
            return self.scaffold_custom(description=description or "", path=path, name=name)

        return self.scaffold(
            template=template,
            path=path,
            name=name,
            options=options or {},
            description=description,
            auto_install=auto_install,
        )

    def list_templates(self) -> ToolResult:
        """Return available templates."""
        lines = []
        for key, info in TEMPLATES.items():
            lines.append(f"  {key:14s} — {info['label']}")
            lines.append(f"  {'':14s}   {info['description']}")
        summary = "Available templates:\n\n" + "\n".join(lines)
        summary += "\n\nUsage: scaffold(template='nextjs', path='./my-app')"
        summary += "\n       scaffold(template='custom', description='a blog with auth')"
        return ToolResult(success=True, result=summary)

    def scaffold(
        self,
        template: str,
        path: str = ".",
        name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        auto_install: bool = False,
    ) -> ToolResult:
        """Generate a project from a built-in template.

        Args:
            template: Template key (nextjs, react-vite, fastapi, express, static, extension).
            path: Target directory. Created if it doesn't exist.
            name: Project name (defaults to directory name).
            options: Extra options dict (e.g. {"description": "My cool app"}).
            description: Shorthand for options["description"].
            auto_install: Run dependency install after scaffolding.
        """
        template = template.lower().strip()
        if template not in _GENERATORS:
            available = ", ".join(_GENERATORS.keys())
            return ToolResult(
                success=False,
                error=f"Unknown template '{template}'. Available: {available}",
            )

        target = Path(path).resolve()
        target.mkdir(parents=True, exist_ok=True)

        project_name = name or target.name or "my-project"
        opts = dict(options or {})
        if description:
            opts.setdefault("description", description)

        try:
            file_map = _GENERATORS[template](project_name, opts)
            created = _write_files(target, file_map)
        except Exception as exc:
            logger.exception("Scaffold generation failed for template=%s", template)
            return ToolResult(success=False, error=f"Generation failed: {exc}")

        result_info: Dict[str, Any] = {
            "template": template,
            "name": project_name,
            "path": str(target),
            "files_created": len(created),
            "files": created,
        }

        # Auto-install
        if auto_install:
            install_result = self._auto_install(template, target)
            result_info["install"] = install_result

        label = TEMPLATES[template]["label"]
        summary = (
            f"Scaffolded '{project_name}' with {label} template.\n"
            f"  Path: {target}\n"
            f"  Files: {len(created)}"
        )
        result_info["summary"] = summary

        return ToolResult(success=True, result=result_info)

    def scaffold_custom(
        self,
        description: str,
        path: str = ".",
        name: Optional[str] = None,
    ) -> ToolResult:
        """AI-powered scaffolding — describe your project and Aura generates it.

        Args:
            description: Natural language description of the project.
            path: Target directory.
            name: Project name (defaults to directory name).
        """
        if not description or not description.strip():
            return ToolResult(
                success=False,
                error="A project description is required for custom scaffolding. "
                      "Example: 'a blog with auth and markdown support'",
            )

        if self._brain is None:
            return ToolResult(
                success=False,
                error="Custom scaffolding requires Aura's brain. No brain instance available. "
                      "Use a built-in template instead: nextjs, react-vite, fastapi, express, static, extension.",
            )

        target = Path(path).resolve()
        target.mkdir(parents=True, exist_ok=True)
        project_name = name or target.name or "my-project"

        prompt = _CUSTOM_SCAFFOLD_PROMPT + description.strip()
        if project_name != "my-project":
            prompt += f"\n\nProject name: {project_name}"

        # Call the brain
        try:
            raw = self._brain._quick_generate(
                prompt,
                system="You are a senior full-stack developer. Return ONLY valid JSON.",
                temperature=0.3,
            )
        except Exception as exc:
            logger.exception("Brain call failed for custom scaffold")
            return ToolResult(success=False, error=f"AI generation failed: {exc}")

        # Parse the JSON from the response
        try:
            parsed = self._extract_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse AI scaffold response: %s", exc)
            return ToolResult(
                success=False,
                error=f"AI returned invalid JSON. Try rephrasing your description. Error: {exc}",
            )

        file_map = parsed.get("files", {})
        if not file_map:
            return ToolResult(
                success=False,
                error="AI generated an empty file set. Try a more specific description.",
            )

        try:
            created = _write_files(target, file_map)
        except Exception as exc:
            logger.exception("Failed to write custom scaffold files")
            return ToolResult(success=False, error=f"Failed to write files: {exc}")

        result_info: Dict[str, Any] = {
            "template": "custom",
            "stack": parsed.get("stack", "custom"),
            "name": project_name,
            "path": str(target),
            "files_created": len(created),
            "files": created,
            "summary": parsed.get("summary", f"Generated {len(created)} files for: {description}"),
        }

        post_install = parsed.get("post_install")
        if post_install:
            result_info["post_install_command"] = post_install

        return ToolResult(success=True, result=result_info)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_json(self, text: str) -> dict:
        """Extract a JSON object from LLM output, handling markdown fences."""
        text = text.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = text.index("\n")
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Find the JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in response")

        return json.loads(text[start:end + 1])

    def _auto_install(self, template: str, target: Path) -> Dict[str, Any]:
        """Run package install for the given template."""
        if template in ("nextjs", "react-vite", "express"):
            cmd = "npm install"
        elif template == "fastapi":
            cmd = f"{sys.executable} -m pip install -r requirements.txt"
        else:
            return {"skipped": True, "reason": "No dependencies to install"}

        import shlex
        try:
            proc = subprocess.run(
                shlex.split(cmd),
                shell=False,
                cwd=str(target),
                capture_output=True,
                text=True,
                timeout=120,
            )
            return {
                "command": cmd,
                "success": proc.returncode == 0,
                "stdout": proc.stdout[-500:] if proc.stdout else "",
                "stderr": proc.stderr[-500:] if proc.stderr else "",
            }
        except subprocess.TimeoutExpired:
            return {"command": cmd, "success": False, "error": "Install timed out (120s)"}
        except Exception as exc:
            return {"command": cmd, "success": False, "error": str(exc)}

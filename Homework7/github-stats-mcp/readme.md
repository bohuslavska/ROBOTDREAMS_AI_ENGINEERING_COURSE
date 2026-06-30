# GitHub Stats MCP

An MCP (Model Context Protocol) server that exposes GitHub profile analytics to Claude Desktop. Built with Python and the official `mcp` package, using stdio transport.

## Tools & Resources

### Tools

#### `analyze_profile` _(required argument)_

Analyze a GitHub user's profile statistics.

| Parameter  | Type   | Required | Default |
|------------|--------|----------|---------|
| `username` | string | Yes      | —       |

Returns: username, name, followers, following, public repo count, total stars, most-starred repo.

---

#### `analyze_languages` _(required + optional argument)_

Analyze programming languages used across a user's repositories.

| Parameter  | Type    | Required | Default |
|------------|---------|----------|---------|
| `username` | string  | Yes      | —       |
| `top_n`    | integer | No       | `5`     |

Returns: username and a ranked map of languages → repo count.

---

#### `get_trending_repos` _(optional arguments)_

Find trending repositories created within the last N months.

| Parameter  | Type    | Required | Default    |
|------------|---------|----------|------------|
| `language` | string  | No       | `"Python"` |
| `months`   | integer | No       | `3`        |

Returns: list of repositories with name, star count, URL, and description.

---

#### `recommend_repositories` _(required + optional argument)_

Recommend popular repositories based on the user's **niche** — derived from the most common topics across their repositories (e.g. `machine-learning`, `cli-tool`, `web-scraping`). Falls back to dominant programming language if no topics are found.

| Parameter  | Type    | Required | Default |
|------------|---------|----------|---------|
| `username` | string  | Yes      | —       |
| `top_n`    | integer | No       | `5`     |

Returns: detected niche (topics or language), niche type, and up to `top_n` recommended repositories with star counts, URLs, descriptions, and topics.

---

### Resources

#### `github://summary/{username}`

Returns a plain-text summary of a GitHub user's profile (name, followers, following, public repos, total stars, top repository). Fetched on demand via the resource URI.

---

## Setup

### Prerequisites

- Python 3.10+
- Claude Desktop (macOS)

### 1. Clone the repository

```bash
git clone <repo-url>
cd github-stats-mcp
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Verify the server starts

```bash
.venv/bin/python server.py
```

You should see the MCP server start without errors (it waits for stdio input — `Ctrl+C` to stop).

### 4. Configure Claude Desktop

Open (or create) `~/Library/Application Support/Claude/claude_desktop_config.json` and **merge** the following block into the `mcpServers` object:

```json
{
  "mcpServers": {
    "github-stats": {
      "command": "/absolute/path/to/github-stats-mcp/.venv/bin/python",
      "args": [
        "/absolute/path/to/github-stats-mcp/server.py"
      ]
    }
  }
}
```

> **Note:** Use the absolute paths from your machine. The `command` must point to the Python binary **inside** the `.venv` so all dependencies are available.

### 5. Restart Claude Desktop

Quit and relaunch Claude Desktop. You should see the GitHub Stats tools available in the tool picker (hammer icon).

---

## Example Dialogs

### Dialog 1 — Analyze a profile

**User:** Analyze GitHub profile torvalds

**Claude:** *(calls `analyze_profile("torvalds")` and `analyze_languages("torvalds")`)* Here's a snapshot of torvalds' GitHub profile — 308,482 followers, 12 public repos, 249,762 total stars. Most-starred repo: `linux` with 237,259 stars. Language breakdown: C (10 repos), OpenSCAD (1), C++ (1).

---

### Dialog 2 — Trending repositories

**User:** Show me trending Python repositories from the last 3 months

**Claude:** *(calls `get_trending_repos(language="Python", months=3)`)* Here are the top trending Python repos from the last 3 months: odysseus (76,337 ⭐), graphify (70,690 ⭐), mempalace (56,168 ⭐), and more.

---

### Dialog 3 — Repository recommendations by niche

**User:** Recommend me repositories similar to Andrej Karpathy

**Claude:** *(calls `recommend_repositories("karpathy")`)* Found his niche: deep learning / arxiv / flask-style projects. Based on that, recommended repos include `rasbt/LLMs-from-scratch`, `huggingface/transformers`, and other educational deep learning repos.

---

## Known Limitations

- **GitHub API rate limits:** Unauthenticated requests are limited to 60 per hour. If you hit the limit, requests will start failing with HTTP 403. Add a `GITHUB_TOKEN` env variable and pass it in the `Authorization` header to raise the limit to 5 000/hour.
- **Public repos only:** The server only reads public repository data via GitHub's public API. Private repos are not accessible.
- **Language detection is per-repo, not by lines of code:** GitHub assigns one "primary language" per repo, so multi-language projects are only counted once under their dominant language.
- **`recommend_repositories` may return fewer than 5 results** if the user already owns most of the top repos for that language.
- **No caching:** Every tool call makes fresh HTTP requests. Repeated calls for the same user will re-fetch all data.

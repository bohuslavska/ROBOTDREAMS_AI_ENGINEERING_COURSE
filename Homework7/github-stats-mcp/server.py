from collections import Counter
from datetime import datetime, timedelta

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("GitHub Stats MCP")

GITHUB_HEADERS = {"Accept": "application/vnd.github+json"}


def get_user(username: str) -> dict:
    response = requests.get(
        f"https://api.github.com/users/{username}",
        timeout=30,
        headers=GITHUB_HEADERS,
    )
    response.raise_for_status()
    return response.json()


def get_repos(username: str) -> list:
    response = requests.get(
        f"https://api.github.com/users/{username}/repos?per_page=100",
        timeout=30,
        headers=GITHUB_HEADERS,
    )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def analyze_profile(username: str) -> dict:
    """
    Analyze a GitHub user's profile statistics.

    Args:
        username: GitHub username (required)
    """
    user = get_user(username)
    repos = get_repos(username)

    total_stars = sum(repo["stargazers_count"] for repo in repos)

    most_starred_repo = None
    if repos:
        best_repo = max(repos, key=lambda repo: repo["stargazers_count"])
        most_starred_repo = {
            "name": best_repo["name"],
            "stars": best_repo["stargazers_count"],
        }

    return {
        "username": user["login"],
        "name": user.get("name"),
        "followers": user["followers"],
        "following": user["following"],
        "public_repos": user["public_repos"],
        "total_stars": total_stars,
        "most_starred_repo": most_starred_repo,
    }


@mcp.tool()
def analyze_languages(
    username: str,
    top_n: int = 5,
) -> dict:
    """
    Analyze programming languages used across a GitHub user's repositories.

    Args:
        username: GitHub username (required)
        top_n: Number of top languages to return, default is 5 (optional)
    """
    repos = get_repos(username)
    counter = Counter()

    for repo in repos:
        language = repo.get("language")
        if language:
            counter[language] += 1

    return {
        "username": username,
        "top_languages": dict(counter.most_common(top_n)),
    }


@mcp.tool()
def get_trending_repos(
    language: str = "Python",
    months: int = 3,
) -> dict:
    """
    Find trending repositories created within the last N months.

    Args:
        language: Programming language filter, default is Python (optional)
        months: How many months back to search, default is 3 (optional)
    """
    since = (datetime.utcnow() - timedelta(days=months * 30)).strftime("%Y-%m-%d")
    response = requests.get(
        "https://api.github.com/search/repositories",
        params={
            "q": f"language:{language} created:>{since}",
            "sort": "stars",
            "order": "desc",
            "per_page": 10,
        },
        headers=GITHUB_HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    repos = []
    for repo in response.json()["items"]:
        repos.append(
            {
                "name": repo["full_name"],
                "stars": repo["stargazers_count"],
                "url": repo["html_url"],
                "description": repo.get("description"),
            }
        )

    return {
        "language": language,
        "months": months,
        "repositories": repos,
    }


@mcp.tool()
def recommend_repositories(username: str, top_n: int = 5) -> dict:
    """
    Recommend popular repositories based on the user's niche — derived from
    the most common topics across their repositories. Falls back to dominant
    programming language if no topics are found.

    Args:
        username: GitHub username (required)
        top_n: Number of recommendations to return, default is 5 (optional)
    """
    repos = get_repos(username)
    owned_repos = {repo["full_name"].lower() for repo in repos}
    topic_counter: Counter = Counter()
    languages: dict[str, int] = {}

    for repo in repos:
        for topic in repo.get("topics", []):
            topic_counter[topic] += 1
        language = repo.get("language")
        if language:
            languages[language] = languages.get(language, 0) + 1

    if topic_counter:
        top_topics = [t for t, _ in topic_counter.most_common(3)]
        query = " ".join(f"topic:{t}" for t in top_topics)
        niche_label = ", ".join(top_topics)
        niche_type = "topics"
    elif languages:
        dominant_language = max(languages, key=languages.get)  # type: ignore[arg-type]
        query = f"language:{dominant_language}"
        niche_label = dominant_language
        niche_type = "language"
    else:
        return {"username": username, "niche": None, "recommendations": []}

    response = requests.get(
        "https://api.github.com/search/repositories",
        params={
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 30,
        },
        headers=GITHUB_HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    recommendations = []
    for repo in response.json()["items"]:
        if repo["full_name"].lower() in owned_repos:
            continue
        recommendations.append(
            {
                "name": repo["full_name"],
                "stars": repo["stargazers_count"],
                "url": repo["html_url"],
                "description": repo.get("description"),
                "topics": repo.get("topics", []),
            }
        )
        if len(recommendations) == top_n:
            break

    return {
        "username": username,
        "niche": niche_label,
        "niche_type": niche_type,
        "recommendations": recommendations,
    }


@mcp.resource("github://summary/{username}")
def github_summary(username: str) -> str:
    """GitHub profile summary as a plain-text resource."""
    user = get_user(username)
    repos = get_repos(username)

    total_stars = sum(repo["stargazers_count"] for repo in repos)
    top_repo = "N/A"
    if repos:
        top_repo = max(repos, key=lambda repo: repo["stargazers_count"])["name"]

    return f"""GitHub Summary for {user["login"]}
====================================
Name:               {user.get("name") or "—"}
Followers:          {user["followers"]}
Following:          {user["following"]}
Public Repos:       {user["public_repos"]}
Total Stars:        {total_stars}
Top Repository:     {top_repo}
"""


if __name__ == "__main__":
    mcp.run()

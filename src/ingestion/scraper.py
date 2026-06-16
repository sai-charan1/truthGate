import requests
import json
import time
import re
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs"
OUTPUT_DIR = Path("data/raw")

DOC_PATHS = [
    ("index", "/index.md"),
    ("fastapi-people", "/fastapi-people.md"),
    ("python-types", "/python-types.md"),
    ("async", "/async.md"),
    ("tutorial-intro", "/tutorial/index.md"),
    ("path-params", "/tutorial/path-params.md"),
    ("query-params", "/tutorial/query-params.md"),
    ("body", "/tutorial/body.md"),
    ("query-params-str-validations", "/tutorial/query-params-str-validations.md"),
    ("path-params-numeric-validations", "/tutorial/path-params-numeric-validations.md"),
    ("query-param-models", "/tutorial/query-param-models.md"),
    ("body-multiple-params", "/tutorial/body-multiple-params.md"),
    ("body-fields", "/tutorial/body-fields.md"),
    ("body-nested-models", "/tutorial/body-nested-models.md"),
    ("schema-extra-example", "/tutorial/schema-extra-example.md"),
    ("extra-data-types", "/tutorial/extra-data-types.md"),
    ("cookie-params", "/tutorial/cookie-params.md"),
    ("cookie-param-models", "/tutorial/cookie-param-models.md"),
    ("header-params", "/tutorial/header-params.md"),
    ("header-param-models", "/tutorial/header-param-models.md"),
    ("response-model", "/tutorial/response-model.md"),
    ("extra-models", "/tutorial/extra-models.md"),
    ("response-status-code", "/tutorial/response-status-code.md"),
    ("request-forms", "/tutorial/request-forms.md"),
    ("request-form-models", "/tutorial/request-form-models.md"),
    ("request-files", "/tutorial/request-files.md"),
    ("request-forms-and-files", "/tutorial/request-forms-and-files.md"),
    ("handling-errors", "/tutorial/handling-errors.md"),
    ("path-operation-configuration", "/tutorial/path-operation-configuration.md"),
    ("encoder", "/tutorial/encoder.md"),
    ("body-updates", "/tutorial/body-updates.md"),
    ("dependencies-intro", "/tutorial/dependencies/index.md"),
    ("classes-as-dependencies", "/tutorial/dependencies/classes-as-dependencies.md"),
    ("sub-dependencies", "/tutorial/dependencies/sub-dependencies.md"),
    ("dependencies-in-path-operation-decorators", "/tutorial/dependencies/dependencies-in-path-operation-decorators.md"),
    ("global-dependencies", "/tutorial/dependencies/global-dependencies.md"),
    ("dependencies-with-yield", "/tutorial/dependencies/dependencies-with-yield.md"),
    ("security-intro", "/tutorial/security/index.md"),
    ("security-first-steps", "/tutorial/security/first-steps.md"),
    ("get-current-user", "/tutorial/security/get-current-user.md"),
    ("simple-oauth2", "/tutorial/security/simple-oauth2.md"),
    ("oauth2-jwt", "/tutorial/security/oauth2-jwt.md"),
    ("middleware", "/tutorial/middleware.md"),
    ("cors", "/tutorial/cors.md"),
    ("sql-databases", "/tutorial/sql-databases.md"),
    ("bigger-applications", "/tutorial/bigger-applications.md"),
    ("background-tasks", "/tutorial/background-tasks.md"),
    ("metadata", "/tutorial/metadata.md"),
    ("static-files", "/tutorial/static-files.md"),
    ("testing", "/tutorial/testing.md"),
    ("debugging", "/tutorial/debugging.md"),
    ("advanced-intro", "/advanced/index.md"),
    ("path-operation-advanced-configuration", "/advanced/path-operation-advanced-configuration.md"),
    ("additional-status-codes", "/advanced/additional-status-codes.md"),
    ("response-directly", "/advanced/response-directly.md"),
    ("custom-response", "/advanced/custom-response.md"),
    ("additional-responses", "/advanced/additional-responses.md"),
    ("response-cookies", "/advanced/response-cookies.md"),
    ("response-headers", "/advanced/response-headers.md"),
    ("advanced-dependencies", "/advanced/advanced-dependencies.md"),
    ("advanced-security-intro", "/advanced/security/index.md"),
    ("using-request-directly", "/advanced/using-request-directly.md"),
    ("dataclasses", "/advanced/dataclasses.md"),
    ("advanced-middleware", "/advanced/middleware.md"),
    ("templates", "/advanced/templates.md"),
    ("websockets", "/advanced/websockets.md"),
    ("events", "/advanced/events.md"),
    ("testing-dependencies", "/advanced/testing-dependencies.md"),
    ("settings", "/advanced/settings.md"),
    ("openapi-callbacks", "/advanced/openapi-callbacks.md"),
    ("sub-applications", "/advanced/sub-applications.md"),
    ("behind-a-proxy", "/advanced/behind-a-proxy.md"),
    ("deployment-intro", "/deployment/index.md"),
    ("deployment-versions", "/deployment/versions.md"),
    ("deployment-https", "/deployment/https.md"),
    ("deployment-manually", "/deployment/manually.md"),
    ("deployment-concepts", "/deployment/concepts.md"),
    ("deployment-docker", "/deployment/docker.md"),
    ("deployment-server-workers", "/deployment/server-workers.md"),
    ("benchmarks", "/benchmarks.md"),
    ("alternatives", "/alternatives.md"),
    ("history-design-future", "/history-design-future.md"),
    ("contributing", "/contributing.md"),
    ("project-generation", "/project-generation.md"),
]


def clean_markdown(md: str) -> str:
    md = re.sub(r'{\*[^}]+\*}', '', md)
    md = re.sub(r'---.*?---', '', md, flags=re.DOTALL)
    md = re.sub(r'```[\w]*\n.*?```', lambda m: m.group(0), md, flags=re.DOTALL)
    md = re.sub(r'!?\[([^\]]*)\]\([^\)]*\)', r'\1', md)
    md = re.sub(r'<[^>]+>', '', md)
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md.strip()


def parse_sections(md: str, title: str) -> list:
    lines = md.split('\n')
    sections = []
    current_heading = title
    current_lines = []

    for line in lines:
        heading_match = re.match(r'^#{1,3}\s+(.+?)(\s*\{[^}]*\})?$', line)
        if heading_match:
            text = ' '.join(current_lines).strip()
            if text and len(text.split()) > 10:
                sections.append({"heading": current_heading, "text": text})
            current_heading = heading_match.group(1).strip()
            current_lines = []
        else:
            stripped = line.strip()
            if stripped:
                current_lines.append(stripped)

    text = ' '.join(current_lines).strip()
    if text and len(text.split()) > 10:
        sections.append({"heading": current_heading, "text": text})

    return sections


def run_scraper():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "TruthGate-RAG/1.0"
    all_pages = []

    for slug, path in DOC_PATHS:
        url = RAW_BASE + path
        page_url = "https://fastapi.tiangolo.com" + path.replace(".md", "/").replace("/index/", "/")
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"  SKIP {url} -> {resp.status_code}")
                continue
            md = resp.text
            cleaned = clean_markdown(md)
            title_match = re.search(r'^#\s+(.+?)(\s*\{[^}]*\})?$', cleaned, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else slug.replace("-", " ").title()
            sections = parse_sections(cleaned, title)
            if sections:
                all_pages.append({"url": page_url, "title": title, "sections": sections})
                print(f"  OK  {slug} -> {len(sections)} sections")
            time.sleep(0.2)
        except Exception as e:
            print(f"  ERR {url}: {e}")

    out_file = OUTPUT_DIR / "fastapi_docs.json"
    with open(out_file, "w") as f:
        json.dump(all_pages, f, indent=2)
    total_sections = sum(len(p["sections"]) for p in all_pages)
    print(f"\nScraped {len(all_pages)} pages, {total_sections} sections -> {out_file}")
    return all_pages


if __name__ == "__main__":
    run_scraper()

"""
Deakin University Course HTML → JSON Extractor
Extracts course information from Deakin unit HTML pages into a structured JSON format.
"""

import json
import re
import sys
from pathlib import Path
from bs4 import BeautifulSoup


def get_text_clean(element) -> str:
    """Return stripped inner text of a BeautifulSoup element, or empty string."""
    if element is None:
        return ""
    return element.get_text(separator=" ", strip=True)


def extract_meta(soup: BeautifulSoup, name: str) -> str | None:
    """Extract a <meta name='...'> content value."""
    tag = soup.find("meta", attrs={"name": name})
    if tag and tag.get("content", "").strip():
        return tag["content"].strip()
    return None


def extract_code_and_name(soup: BeautifulSoup) -> tuple[str, str]:
    """Extract unit code and name from the <h1> heading."""
    h1 = soup.find("h1")
    if not h1:
        return "", ""
    text = h1.get_text(strip=True)
    # Pattern: "DAI001 - Academic Integrity and Respect at Deakin"
    match = re.match(r"^([A-Z0-9]+)\s*[-–]\s*(.+)$", text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return text, ""


def extract_description(soup: BeautifulSoup) -> str:
    """Extract the unit description from the Content section."""
    content_div = soup.find("div", class_="content")
    if not content_div:
        return ""

    # Find the <h3>Content</h3> tag and grab the next <p>
    for h3 in content_div.find_all("h3"):
        if h3.get_text(strip=True).lower() == "content":
            # Collect all <p> tags until the next heading
            parts = []
            for sibling in h3.next_siblings:
                if sibling.name and sibling.name.startswith("h"):
                    break
                if sibling.name == "p":
                    parts.append(sibling.get_text(strip=True))
            return "\n\n".join(parts)

    # Fallback: meta description
    return extract_meta(soup, "description") or ""


def extract_learning_outcomes(soup: BeautifulSoup) -> list[str]:
    """
    Extract ULO descriptions from the Learning Outcomes table.
    Returns a list of outcome strings.
    """
    outcomes = []
    content_div = soup.find("div", class_="content")
    if not content_div:
        return outcomes

    for h3 in content_div.find_all("h3"):
        if "learning outcome" in h3.get_text(strip=True).lower():
            # Find the next <table>
            table = h3.find_next("table")
            if not table:
                break
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                # Rows have: ULO label | description | GLO alignment
                if len(cells) >= 2:
                    # Second cell is the outcome description
                    text = cells[1].get_text(separator=" ", strip=True)
                    if text:
                        outcomes.append(text)
            break

    return outcomes


def extract_assessment(soup: BeautifulSoup) -> str:
    """
    Extract assessment information as a plain text block.
    Includes the assessment table, hurdle requirements, and any notes.
    """
    content_div = soup.find("div", class_="content")
    if not content_div:
        return ""

    parts = []
    in_assessment = False

    for tag in content_div.children:
        if not hasattr(tag, "name") or not tag.name:
            continue

        # Start capturing at <h3>Assessment</h3>
        if tag.name == "h3" and "assessment" in tag.get_text(strip=True).lower():
            in_assessment = True
            continue

        # Stop at the next h3 that isn't assessment-related
        if in_assessment and tag.name == "h3":
            heading = tag.get_text(strip=True).lower()
            if "hurdle" in heading:
                parts.append(f"\n{tag.get_text(strip=True)}:")
                continue
            else:
                break

        if in_assessment:
            if tag.name == "table":
                # Flatten table rows into readable text
                for row in tag.find_all("tr"):
                    cells = [td.get_text(separator=" ", strip=True) for td in row.find_all(["th", "td"])]
                    if any(cells):
                        parts.append(" | ".join(cells))
            elif tag.name == "p":
                text = tag.get_text(strip=True)
                if text:
                    parts.append(text)

    return "\n".join(parts).strip()


def extract_prerequisites(soup: BeautifulSoup) -> list[str]:
    """
    Extract prerequisites from the meta tag or from unit details table.
    Returns list of unit codes, or empty list if none.
    """
    prereq_str = extract_meta(soup, "dknunitprereq")
    if not prereq_str:
        return []
    # Split comma/semicolon separated codes
    codes = [c.strip() for c in re.split(r"[,;]+", prereq_str) if c.strip()]
    return codes


def extract_nominal_hours(soup: BeautifulSoup) -> float | None:
    """
    Attempt to extract nominal hours from unit details table or workload meta.
    Returns None if not found.
    """
    workload = extract_meta(soup, "dknunitworkload")
    if workload:
        match = re.search(r"(\d+(?:\.\d+)?)", workload)
        if match:
            return float(match.group(1))

    # Try scanning the details table for hour-related text
    content_div = soup.find("div", class_="content")
    if content_div:
        table = content_div.find("table", class_="table")
        if table:
            for row in table.find_all("tr"):
                th = row.find("th")
                td = row.find("td")
                if th and td:
                    label = th.get_text(strip=True).lower()
                    if "hour" in label or "workload" in label:
                        match = re.search(r"(\d+(?:\.\d+)?)", td.get_text())
                        if match:
                            return float(match.group(1))
    return None


def extract_course_info(html_path: str) -> dict:
    """Main extraction function. Returns a dict matching the target JSON schema."""
    html = Path(html_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    code, name = extract_code_and_name(soup)

    return {
        "code": code,
        "name": name,
        "description": extract_description(soup),
        "learning_outcomes": extract_learning_outcomes(soup),
        "assessment_requirements": extract_assessment(soup),
        "nominal_hours": extract_nominal_hours(soup),
        "prerequisites": extract_prerequisites(soup),
    }


def main():
    if len(sys.argv) < 2:
        # Default to the uploaded file for demonstration
        html_path = "/mnt/user-data/uploads/DAI001_-_Academic_Integrity_and_Respect_at_Deakin.html"
    else:
        html_path = sys.argv[1]

    result = extract_course_info(html_path)

    output_path = Path("/mnt/user-data/outputs/course_info.json")
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n✅ Saved to {output_path}")


if __name__ == "__main__":
    main()

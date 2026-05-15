import json
import re
import time
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional

# ──────────────────────────────────────────────
# Known degree structure from La Trobe / OUA
# (year → list of OUA subject slugs)
# ──────────────────────────────────────────────
DEGREE = {
    "code": "LAT-CYS-DEG",
    "name": "Bachelor of Cybersecurity",
}

SUBJECTS_BY_YEAR = {
    1: [
        "la-trobe-university-introduction-to-cybersecurity-ltu-cse1icb",
        "la-trobe-university-cybersecurity-in-practice-ltu-cse1cpr",
        "la-trobe-university-cryptography-and-security-ltu-mat1003",
        "la-trobe-university-network-engineering-fundamentals-ltu-cse1nef",
        "la-trobe-university-object-oriented-programming-fundamentals-ltu-cse1oop",
        "la-trobe-university-inside-information-technology-ltu-cse1iit",
        "la-trobe-university-data-based-critical-thinking-ltu-mat1dct",
        "la-trobe-university-programming-environment-ltu-cse1pen",
    ],
    2: [
        "la-trobe-university-wireless-network-and-security-ltu-cse2win",
        "la-trobe-university-cyber-law-and-policy-ltu-cse2clp",
        "la-trobe-university-data-security-and-information-assurance-ltu-cse2sia",
        "la-trobe-university-human-factors-in-cybersecurity-ltu-cse2hum",
    ],
    3: [
        "la-trobe-university-capstone-project-ltu-cse3cap",
        "la-trobe-university-professional-practices-and-entrepreneurship-in-information-technology-ltu-cse3ppe",
        "la-trobe-university-project-management-ltu-cse3pjm",
        "la-trobe-university-introduction-to-computer-forensics-ltu-cse3cfn",
        "la-trobe-university-intermediate-network-engineering-ltu-cse3ine",
        "la-trobe-university-blockchain-and-cryptocurrencies-ltu-cse3blk",
        "la-trobe-university-introduction-to-penetration-testing-ltu-cse3002",
        "la-trobe-university-network-systems-and-web-security-ltu-cse3nsw",
    ],
}

BASE_URL = "https://www.open.edu.au/subjects"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_page(url: str) -> BeautifulSoup:
    """Fetch a page and return a BeautifulSoup object."""
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def extract_subject_code(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract the subject code from the page heading.
    Format: "Undergraduate | LTU-CSE1ICB | 2026"
    """
    page_text = soup.get_text(separator="\n", strip=True)
    # Match patterns like LTU-CSE1ICB or CSE1ICB
    match = re.search(r'\b(LTU-[A-Z0-9]+)\b', page_text)
    if match:
        return match.group(1)
    return None


def extract_subject_name(soup: BeautifulSoup) -> Optional[str]:
    """Extract the subject name from the h1 tag."""
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return None


def extract_description(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract description — content between Introduction/Overview and Learning outcomes.
    Falls back to the meta description or the tagline under the h1.
    """
    page_text = soup.get_text(separator="\n", strip=True)

    # Try to find content between known intro markers and learning outcomes
    for start_marker in ["Introduction", "Overview", "About this subject", "Description"]:
        start_idx = page_text.find(start_marker)
        end_idx = page_text.find("What you'll learn", start_idx)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            raw = page_text[start_idx + len(start_marker):end_idx].strip()
            lines = [l.strip() for l in raw.split("\n") if l.strip() and len(l.strip()) > 20]
            if lines:
                return " ".join(lines)

    # Fallback: the short tagline that appears just below the h1 / subject code line
    # On OUA pages this is a short paragraph before the study details table
    h1 = soup.find("h1")
    if h1:
        # Walk siblings until we find a paragraph-like element
        for sibling in h1.find_next_siblings():
            text = sibling.get_text(strip=True)
            if text and len(text) > 30 and sibling.name in ["p", "div", "span"]:
                return text

    # Final fallback: meta description
    meta = soup.find("meta", {"name": "description"})
    if meta and meta.get("content"):
        return meta["content"].strip()

    return None


def extract_learning_outcomes(soup: BeautifulSoup) -> List[str]:
    """
    Extract learning outcomes from the 'What you'll learn' section.
    OUA pages list them as a numbered list after this heading.
    """
    outcomes = []

    # Strategy 1: find heading then collect the numbered list below it
    for heading in soup.find_all(["h2", "h3", "h4", "h5", "strong", "b"]):
        heading_text = heading.get_text(strip=True).lower()
        if "what you'll learn" in heading_text or "learning outcomes" in heading_text:
            sibling = heading.find_next_sibling()
            while sibling:
                if sibling.name in ["ol", "ul"]:
                    for li in sibling.find_all("li"):
                        text = li.get_text(strip=True)
                        if text:
                            outcomes.append(text)
                    break
                elif sibling.name in ["h2", "h3", "h4"]:
                    break
                sibling = sibling.find_next_sibling()
            if outcomes:
                break

    # Strategy 2: plain text between "What you'll learn" and next section
    if not outcomes:
        page_text = soup.get_text(separator="\n", strip=True)
        for start_marker in ["What you'll learn", "On successful completion you will be able to:"]:
            start_idx = page_text.find(start_marker)
            if start_idx == -1:
                continue
            end_idx = page_text.find("Topics covered", start_idx)
            if end_idx == -1:
                end_idx = page_text.find("Description", start_idx)
            if end_idx == -1:
                end_idx = start_idx + 2000  # cap at 2000 chars
            section = page_text[start_idx + len(start_marker):end_idx]
            for line in section.split("\n"):
                line = re.sub(r"^\d+\.\s*", "", line.strip()).strip()
                line = re.sub(r"^[-•·]\s*", "", line).strip()
                if line and len(line) > 10:
                    outcomes.append(line)
            if outcomes:
                break

    return outcomes


def extract_assessment(soup: BeautifulSoup) -> Optional[str]:
    """Extract assessment information from the Assessments section."""
    page_text = soup.get_text(separator="\n", strip=True)

    next_sections = ["Textbooks", "About La Trobe", "Entry requirements", "Study load"]
    for next_section in next_sections:
        start_idx = page_text.find("Assessments")
        end_idx = page_text.find(next_section, start_idx)
        if start_idx != -1 and end_idx != -1:
            raw = page_text[start_idx + len("Assessments"):end_idx].strip()
            lines = [l.strip() for l in raw.split("\n") if l.strip() and len(l.strip()) > 10]
            if lines:
                return " ".join(lines)

    return None


def extract_nominal_hours(soup: BeautifulSoup) -> Optional[int]:
    """
    Extract nominal hours from the study load section.
    OUA states 'EFTSL' and '10 to 12 hours of study each week' for 12-week subjects.
    """
    page_text = soup.get_text(separator=" ", strip=True)

    # Look for explicit hours per week
    match = re.search(r"(\d+)\s*to\s*(\d+)\s*hours\s*of\s*study\s*each\s*week", page_text, re.I)
    if match:
        # Return the average hours × 12 weeks
        avg = (int(match.group(1)) + int(match.group(2))) // 2
        return avg * 12

    # Look for duration in weeks
    week_match = re.search(r"Duration\s*(\d+)\s*weeks?", page_text, re.I)
    if week_match:
        return int(week_match.group(1)) * 10  # assume ~10 hrs/week default

    return None


def extract_credit_points(soup: BeautifulSoup) -> Optional[int]:
    """
    Extract credit points from the page.
    OUA pages typically show EFTSL; 0.125 EFTSL = 15 credit points at La Trobe.
    Also looks for explicit credit point mentions.
    """
    page_text = soup.get_text(separator=" ", strip=True)

    # Look for explicit credit points mention
    match = re.search(r"(\d+)\s*credit\s*points?", page_text, re.I)
    if match:
        return int(match.group(1))

    # Derive from EFTSL: 0.125 EFTSL = 15 credit points (La Trobe standard)
    eftsl_match = re.search(r"(\d+\.?\d*)\s*EFTSL", page_text, re.I)
    if eftsl_match:
        eftsl = float(eftsl_match.group(1))
        return int(eftsl * 120)  # 1.0 EFTSL = 120 credit points

    return None


def scrape_subject(slug: str, year: int) -> Optional[Dict]:
    """
    Scrape a single subject page and return its data as a dict.
    Returns None if the page cannot be fetched or parsed.
    """
    url = f"{BASE_URL}/{slug}"
    print(f"  Fetching: {url}")

    try:
        soup = fetch_page(url)
        return {
            "code":                    extract_subject_code(soup),
            "name":                    extract_subject_name(soup),
            "description":             extract_description(soup),
            "study_level":             f"University_Year_{year}",
            "learning_outcomes":       extract_learning_outcomes(soup),
            "assessment":              extract_assessment(soup),
            "nominal_hours":           extract_nominal_hours(soup),
            "credit_points":           extract_credit_points(soup),
            "year":                    year,
        }
    except Exception as e:
        print(f"  ERROR scraping {slug}: {e}")
        return None


def build_degree_json() -> Dict:
    """Scrape all subjects and assemble the degree JSON."""
    units = []

    for year, slugs in SUBJECTS_BY_YEAR.items():
        print(f"\n── Year {year} ──")
        for slug in slugs:
            unit = scrape_subject(slug, year)
            if unit is not None:  # Only add successfully fetched units
                units.append(unit)
            time.sleep(1)  # polite delay between requests

    return {
        "code":  DEGREE["code"],
        "name":  DEGREE["name"],
        "courses": units,
    }


def save_to_json(data: Dict, output_path: str = "cybersecurity_degree.json"):
    """Save the result to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


def main():
    print("Scraping La Trobe Bachelor of Cybersecurity...\n")
    degree = build_degree_json()

    print("\nExtracted Degree JSON:")
    print(json.dumps(degree, indent=2, ensure_ascii=False))

    save_to_json(degree)


if __name__ == "__main__":
    main()

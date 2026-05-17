import json
import re
import time
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from typing import Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────
# Full subject list sourced directly from La Trobe handbook
# ──────────────────────────────────────────────────────────────────────
# SUBJECTS = [
#     # (subject_code, subject_name, year, credit_points, oua_slug or None)
#     # None = not on OUA, will fall back to La Trobe handbook
#     ("LTU0AIM", "Academic Integrity Module",                                              1,  0,  None),
#     ("ABS0WOM", "Wominjeka La Trobe: Indigenous Cultural Literacy for Higher Education",  1,  0,  None),
#     ("MAT1003", "Cryptography and Security",                                              1, 15, "la-trobe-university-cryptography-and-security-ltu-mat1003"),
#     ("CSE1CPR", "Cybersecurity in Practice",                                              1, 15, "la-trobe-university-cybersecurity-in-practice-ltu-cse1cpr"),
#     ("STA1DCT", "Data-Based Critical Thinking",                                           1, 15, "la-trobe-university-data-based-critical-thinking-ltu-sta1dct"),
#     ("CSE1IIT", "Inside Information Technology",                                          1, 15, "la-trobe-university-inside-information-technology-ltu-cse1iit"),
#     ("CSE1ICB", "Introduction to Cybersecurity",                                          1, 15, "la-trobe-university-introduction-to-cybersecurity-ltu-cse1icb"),
#     ("CSE2NEF", "Network Engineering Fundamentals",                                       1, 15, "la-trobe-university-network-engineering-fundamentals-ltu-cse2nef"),
#     ("CSE1OOF", "Object-Oriented Programming Fundamentals",                               1, 15, "la-trobe-university-object-oriented-programming-fundamentals-ltu-cse1oof"),
#     ("CSE1PE",  "Programming Environment",                                                1, 15, "la-trobe-university-programming-environment-ltu-cse1pe"),
#     ("LAW2CLP", "Cyber Law and Policy",                                                   2, 15, "la-trobe-university-cyber-law-and-policy-ltu-law2clp"),
#     ("CSE2SIA", "Data Security and Information Assurance",                                2, 15, "la-trobe-university-data-security-and-information-assurance-ltu-cse2sia"),
#     ("CSE2HUM", "Human Factors in Cybersecurity",                                         2, 15, "la-trobe-university-human-factors-in-cybersecurity-ltu-cse2hum"),
#     ("CSE2WIN", "Wireless Network and Security",                                          2, 15, "la-trobe-university-wireless-network-and-security-ltu-cse2win"),
#     ("CSE3BCC", "Blockchain and Cryptocurrencies",                                        3, 15,  None),  # not on OUA, uses handbook
#     ("CSE3INE", "Intermediate Network Engineering",                                       3, 15, "la-trobe-university-intermediate-network-engineering-ltu-cse3ine"),
#     ("CSE3CFN", "Introduction to Computer Forensics",                                     3, 15, "la-trobe-university-introduction-to-computer-forensics-ltu-cse3cfn"),
#     ("CSE3002", "Introduction to Penetration Testing",                                    3, 15, "la-trobe-university-introduction-to-penetration-testing-ltu-cse3002"),
#     ("CSE3NSW", "Network Systems and Web Security",                                       3, 15, "la-trobe-university-network-systems-and-web-security-ltu-cse3nsw"),
#     ("CSE3PPE", "Professional Practices and Entrepreneurship in Information Technology",  3, 15, "la-trobe-university-professional-practices-and-entrepreneurship-in-information-technology-ltu-cse3ppe"),
#     ("CSE3PM",  "Project Management",                                                     3, 15, "la-trobe-university-project-management-ltu-cse3pm"),
# ]

SUBJECTS = [
    # (subject_code, subject_name, year, credit_points, oua_slug or None)
    # None = not on OUA, will fall back to La Trobe handbook
    ("LTU0AIM", "Academic Integrity Module",                                              1,  0,  None),
    ("ABS0WOM", "Wominjeka La Trobe: Indigenous Cultural Literacy for Higher Education",  1,  0,  None),
    ("MAT1003", "Cryptography and Security",                                              1, 15, None),
    ("CSE1CPR", "Cybersecurity in Practice",                                              1, 15, None),
    ("STA1DCT", "Data-Based Critical Thinking",                                           1, 15, None),
    ("CSE1IIT", "Inside Information Technology",                                          1, 15, None),
    ("CSE1ICB", "Introduction to Cybersecurity",                                          1, 15, None),
    ("CSE2NEF", "Network Engineering Fundamentals",                                       1, 15, None),
    ("CSE1OOF", "Object-Oriented Programming Fundamentals",                               1, 15, None),
    ("CSE1PE",  "Programming Environment",                                                1, 15, None),
    ("LAW2CLP", "Cyber Law and Policy",                                                   2, 15, None),
    ("CSE2SIA", "Data Security and Information Assurance",                                2, 15, None),
    ("CSE2HUM", "Human Factors in Cybersecurity",                                         2, 15, None),
    ("CSE2WIN", "Wireless Network and Security",                                          2, 15, None),
    ("CSE3BCC", "Blockchain and Cryptocurrencies",                                        3, 15,  None),  # not on OUA, uses handbook
    ("CSE3INE", "Intermediate Network Engineering",                                       3, 15, None),
    ("CSE3CFN", "Introduction to Computer Forensics",                                     3, 15, None),
    ("CSE3002", "Introduction to Penetration Testing",                                    3, 15, None),
    ("CSE3NSW", "Network Systems and Web Security",                                       3, 15, None),
    ("CSE3PPE", "Professional Practices and Entrepreneurship in Information Technology",  3, 15, None),
    ("CSE3PM",  "Project Management",                                                     3, 15, None),
]

OUA_BASE      = "https://www.open.edu.au/subjects"
HANDBOOK_BASE = "https://handbook.latrobe.edu.au/subjects/2026"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ══════════════════════════════════════════════════════════════════════
# 1. PAGE FETCHERS
# ══════════════════════════════════════════════════════════════════════

def fetch_page(url: str) -> BeautifulSoup:
    """Fetch a static page using requests and return a BeautifulSoup object."""
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def fetch_page_playwright(url: str) -> BeautifulSoup:
    """
    Fetch a JavaScript-rendered page using Playwright.
    Waits for network idle then an extra 2s for lazy content.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=HEADERS["User-Agent"])
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2000)
        html = page.content()
        browser.close()
    return BeautifulSoup(html, "html.parser")


# ══════════════════════════════════════════════════════════════════════
# 2. HELPERS
# ══════════════════════════════════════════════════════════════════════

def text_between(text: str, start: str, *ends) -> Optional[str]:
    """Return text between start marker and the first matching end marker."""
    s = text.find(start)
    if s == -1:
        return None
    s += len(start)
    for end in ends:
        e = text.find(end, s)
        if e != -1:
            return text[s:e].strip()
    return text[s:].strip()


def clean_lines(raw: str, min_len: int = 10) -> List[str]:
    """Split by newline, strip bullets/numbers, drop short lines."""
    lines = []
    for line in raw.split("\n"):
        line = re.sub(r"^[\d]+[.)]\s*", "", line.strip())
        line = re.sub(r"^[-•·]\s*", "", line).strip()
        if line and len(line) >= min_len:
            lines.append(line)
    return lines


# ══════════════════════════════════════════════════════════════════════
# 3. OUA FIELD EXTRACTORS
# ══════════════════════════════════════════════════════════════════════

def extract_description(soup: BeautifulSoup) -> Optional[str]:
    page = soup.get_text(separator="\n", strip=True)
    for start in ["Introduction", "Overview", "About this subject", "Description"]:
        raw = text_between(page, start, "What you'll learn", "Learning outcomes", "Topics covered")
        if raw:
            lines = [l for l in clean_lines(raw) if len(l) > 20]
            if lines:
                return " ".join(lines)
    meta = soup.find("meta", {"name": "description"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    return None


def extract_learning_outcomes(soup: BeautifulSoup) -> List[str]:
    outcomes = []
    # Strategy 1: heading → next <ol>/<ul>
    for heading in soup.find_all(["h2", "h3", "h4", "h5", "strong", "b"]):
        if re.search(r"what you.ll learn|learning outcomes?", heading.get_text(), re.I):
            sibling = heading.find_next_sibling()
            while sibling:
                if sibling.name in ["ol", "ul"]:
                    outcomes = [li.get_text(strip=True) for li in sibling.find_all("li") if li.get_text(strip=True)]
                    break
                if sibling.name in ["h2", "h3", "h4"]:
                    break
                sibling = sibling.find_next_sibling()
            if outcomes:
                return outcomes
    # Strategy 2: plain-text extraction
    page = soup.get_text(separator="\n", strip=True)
    for start in ["What you'll learn", "On successful completion you will be able to:"]:
        raw = text_between(page, start, "Topics covered", "Description", "Assessments", "Entry requirements")
        if raw:
            outcomes = clean_lines(raw)
            if outcomes:
                return outcomes
    return outcomes


def extract_assessment(soup: BeautifulSoup) -> Optional[str]:
    page = soup.get_text(separator="\n", strip=True)
    raw = text_between(page, "Assessments", "Textbooks", "About La Trobe",
                       "Entry requirements", "Study load")
    if raw:
        lines = clean_lines(raw)
        if lines:
            return " ".join(lines)
    return None


def extract_nominal_hours(soup: BeautifulSoup) -> Optional[int]:
    page = soup.get_text(separator=" ", strip=True)
    match = re.search(r"(\d+)\s*to\s*(\d+)\s*hours?\s*of\s*study\s*each\s*week", page, re.I)
    if match:
        avg = (int(match.group(1)) + int(match.group(2))) // 2
        weeks_match = re.search(r"Duration\D*(\d+)\s*weeks?", page, re.I)
        weeks = int(weeks_match.group(1)) if weeks_match else 12
        return avg * weeks
    return None


# ══════════════════════════════════════════════════════════════════════
# 4. HANDBOOK FIELD EXTRACTORS
# ══════════════════════════════════════════════════════════════════════

def extract_description_handbook(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract description from La Trobe handbook pages.
    Content sits between the subject intro and known section markers.
    """
    page = soup.get_text(separator="\n", strip=True)
    raw = text_between(page, "For more content click the Read more button below.",
                       "Graduate capabilities", "Subject intended learning outcomes")
    if not raw:
        raw = text_between(page, "Graduate capabilities",
                           "Subject intended learning outcomes", "On successful completion")
    if raw:
        noise = ["Graduate capabilities", "Subject intended learning outcomes",
                 "Work based learning", "Career ready", "Read more"]
        lines = [l for l in raw.split("\n")
                 if l.strip() and len(l.strip()) > 20
                 and not any(n in l for n in noise)]
        if lines:
            return " ".join(l.strip() for l in lines)
    return None


def extract_learning_outcomes_handbook(soup: BeautifulSoup) -> List[str]:
    """
    Extract learning outcomes from La Trobe handbook pages.
    Filters out UI artifacts like 'keyboard_arrow_down' and nav/footer noise.
    """
    page = soup.get_text(separator="\n", strip=True)
    raw = text_between(page, "On successful completion you will be able to:",
                       "Learning activities", "Enrolment rules", "Work based learning")
    if not raw:
        return []

    noise_exact = {
        "keyboard_arrow_down", "keyboard_arrow_up", "open_in_new",
        "expand_more", "expand_less", "Collapse all", "Expand all",
    }
    noise_partial = [
        "Copyright", "CRICOS", "ABN", "Emergency", "Accessibility",
        "CourseLoop", "Powered by", "La Trobe University",
    ]
    outcomes = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or len(line) < 10:
            continue
        if line in noise_exact:
            continue
        if any(n in line for n in noise_partial):
            continue
        if line.endswith("."):  # outcomes are full sentences
            outcomes.append(line)
    return outcomes


def extract_assessment_handbook(soup: BeautifulSoup) -> Optional[str]:
    """
    Assessment details in the handbook are instance-specific and loaded
    dynamically via a dropdown — not reliably extractable.
    """
    return None


def extract_nominal_hours_handbook(soup: BeautifulSoup) -> Optional[int]:
    """Extract nominal hours from La Trobe handbook pages."""
    page = soup.get_text(separator=" ", strip=True)
    match = re.search(r"(\d+)\s*hours?\s*per\s*week", page, re.I)
    if match:
        hours_pw = int(match.group(1))
        weeks_match = re.search(r"(\d+)\s*weeks?", page, re.I)
        weeks = int(weeks_match.group(1)) if weeks_match else 12
        return hours_pw * weeks
    match = re.search(r"(\d+)\s*(?:total\s*)?(?:contact\s*)?hours", page, re.I)
    if match:
        return int(match.group(1))
    return None


# ══════════════════════════════════════════════════════════════════════
# 5. SUBJECT SCRAPERS
# ══════════════════════════════════════════════════════════════════════

def scrape_from_oua(slug: str) -> Dict:
    """Scrape subject details from Open Universities Australia."""
    url = f"{OUA_BASE}/{slug}"
    print(f"  Fetching OUA: {url}")
    soup = fetch_page(url)
    return {
        "description":       extract_description(soup),
        "learning_outcomes": extract_learning_outcomes(soup),
        "assessment":        extract_assessment(soup),
        "nominal_hours":     extract_nominal_hours(soup),
    }


def scrape_from_handbook(code: str) -> Dict:
    """
    Scrape subject details from the La Trobe handbook using Playwright,
    since the handbook is JavaScript-rendered and requests/bs4 cannot parse it.
    Uses handbook-specific extractors to handle the different page structure.
    """
    url = f"{HANDBOOK_BASE}/{code}"
    print(f"  Falling back to handbook (Playwright): {url}")
    soup = fetch_page_playwright(url)
    return {
        "description":       extract_description_handbook(soup),
        "learning_outcomes": extract_learning_outcomes_handbook(soup),
        "assessment":        extract_assessment_handbook(soup),
        "nominal_hours":     extract_nominal_hours_handbook(soup),
    }


def scrape_subject(code: str, name: str, year: int, credit_points: int, slug: Optional[str]) -> Optional[Dict]:
    """
    Scrape a subject from OUA, falling back to La Trobe handbook if no slug.
    Returns None only if both sources fail.
    """
    base = {
        "code":              code,
        "name":              name,
        "description":       None,
        "study_level":       f"University_Year_{year}",
        "learning_outcomes": [],
        "assessment":        None,
        "nominal_hours":     None,
        "credit_points":     credit_points,
        "year":              year,
    }
    # Try OUA first if slug is available
    if slug is not None:
        try:
            base.update(scrape_from_oua(slug))
            return base
        except Exception as e:
            print(f"  OUA fetch failed for {code}: {e}")

    # Fallback: La Trobe handbook (Playwright)
    try:
        base.update(scrape_from_handbook(code))
        return base
    except Exception as e:
        print(f"  Handbook fetch also failed for {code}: {e}")
        return None  # skip entirely if both sources fail


# ══════════════════════════════════════════════════════════════════════
# 6. DEGREE BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_degree_json() -> Dict:
    courses = []
    for code, name, year, credit_points, slug in SUBJECTS:
        print(f"Processing {code} – {name}")
        unit = scrape_subject(code, name, year, credit_points, slug)
        if unit is not None:
            courses.append(unit)
        if slug is not None:
            time.sleep(1)  # polite delay only for real HTTP requests
    return {
        "code":    "LAT-CYS-DEG",
        "name":    "Bachelor of Cybersecurity",
        "courses": courses,
    }


# ══════════════════════════════════════════════════════════════════════
# 7. I/O
# ══════════════════════════════════════════════════════════════════════

def save_to_json(data: Dict, path: str = "cybersecurity_degree.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {path}")


def main():
    print("Scraping La Trobe Bachelor of Cybersecurity...\n")
    degree = build_degree_json()
    print("\nExtracted Degree JSON:")
    print(json.dumps(degree, indent=2, ensure_ascii=False))
    save_to_json(degree)


if __name__ == "__main__":
    main()

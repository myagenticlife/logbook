"""
OCR service for extracting flight data from logbook images.

Uses Google Gemini as primary method (multimodal LLM for best handwriting
recognition), with Google Vision API and Tesseract as fallbacks.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_SDK_AVAILABLE = True
except ImportError:
    GENAI_SDK_AVAILABLE = False

import requests as _requests



class LogbookOCRService:
    """Service for extracting flight data from logbook images."""

    # Prompt for Gemini to extract structured flight data
    EXTRACTION_PROMPT = """You are analyzing a photo of a physical pilot logbook page.
Extract ALL flight entries visible in the image and return them as a JSON array.

Each flight entry should have these fields (use null if not readable):
- date: MUST always be in "MM/DD/YYYY" format with leading zeros and a 4-digit year. Examples: "05/09/2004", "12/03/2023", "01/15/2010". In handwritten logbooks, the year is often written only once at the top of the page or in a "YEAR" header — you MUST apply that year to every entry on the page. Handwritten dates typically show only month/day (e.g., "1/27" means January 27). Read the month and day digits carefully — a "1" can look like a "7" and vice versa in handwriting. If the year is not visible anywhere on this page, infer it from context: the dates should be sequential and realistic for a pilot logbook (typically 2000-2026). If a date spans multiple days (e.g., "3/26-28"), use the first date. Every date you output MUST have a 4-digit year — never output just "03/10", always "03/10/2004".
- aircraft_model: FAA aircraft type designator (e.g., "DA20", "C172", "PA28", "TBM7", "C206", "SR22", "BE36"). Use standard FAA designators WITHOUT hyphens. Convert full names: "Piper Cub" → "J3", "Cessna 172" → "C172", "Diamond DA20" → "DA20". For simulators/FTDs, write the type followed by "-FTD" (e.g., "DA42-FTD"), or "Frasca-FTD" for Frasca devices. Be careful with OCR-ambiguous characters: Y vs 7, T vs 7, O vs 0, I vs 1, S vs 5. For example, "C2067" is not valid — it should be "C206T" or another real designator. If the same aircraft type appears on multiple rows, ensure consistency.
- aircraft_ident: tail number (e.g., "N636DC", "N95225"). If the same tail number appears on multiple rows, ensure consistency — handwriting OCR often confuses Y/7, T/7, O/0, I/1, S/5, B/8. Pick the most likely real tail number.
- route_from: departure airport ICAO or FAA code (e.g., "BFI", "SEA", "PAE")
- route_to: destination airport ICAO or FAA code (e.g., "BFI", "SEA", "PAE")
- remarks: any remarks or endorsements text. These are aviation shorthand — preserve them as written or use standard aviation abbreviations. Common examples: "ldg" (landing), "appro" or "appr" (approach), "ILS" (instrument landing system), "VOR", "Rwy" (runway), "T&G" or "TnG" (touch and go), "XC" (cross country), "NDB", "GPS", "LOC" (localizer), "RNAV", "steep trns" (steep turns). Instrument approach remarks often look like "ILS 16R PAE" or "RNAV 03 SEZ" (approach type + runway + airport). Do NOT replace aviation terms with non-aviation English words
- sel: single engine land time in decimal hours (the "AIRPLANE SEL" column)
- mel: multi engine land time in decimal hours (the "AIRPLANE MEL" column)
- total_duration: total flight time in decimal hours (e.g., 1.4, 2.2)
- pic: pilot in command time in decimal hours
- sic: second in command time in decimal hours
- dual_recd: dual received time in decimal hours
- dual_given: dual given/flight instructor time in decimal hours
- solo: solo time in decimal hours
- cross_country: cross country time in decimal hours
- night: night time in decimal hours
- actual_inst: actual instrument time in decimal hours
- simulated_inst: simulated instrument (hood) time in decimal hours
- num_inst_app: number of instrument approaches (integer)
- landings_day: number of day landings (integer). For a simple A→B flight this is typically 1 unless it is a training flight with touch-and-goes. Read the digit carefully — "1" is far more common than "2" for a single-leg flight.
- landings_night: number of night landings (integer)
- sim: simulator/FTD time in decimal hours (0 for real flights; only non-zero for FTD/simulator entries)

Important:
- You MUST output one entry for EVERY row of flight data visible in the logbook, even if you cannot read it well. Use empty strings for unreadable text fields and 0 for unreadable numbers. Never skip a row - the user needs a placeholder to fill in manually
- Airport codes MUST be valid real FAA or ICAO identifiers. If the handwriting is ambiguous, infer the most likely real airport code. For example, "BEE" is not a valid code but "BFI" (Boeing Field, Seattle) is. Common codes include: BFI, SEA, PAE, RNT, PWT, OLM, S43, S50, 0S9, HQM, BLI, etc.
- If a route shows multiple stops like "BFI-PAE-BFI", set route_from to the first, route_to to the last, and put the full route (e.g., "BFI-PAE-BFI") at the BEGINNING of the remarks field, followed by any other remarks
- Set numeric fields to 0 if the cell is empty (not null)
- Only extract ACTUAL FLIGHT entries. A real flight row has a date, aircraft, and airports. SKIP any summary/totals rows — these are rows that only contain numbers (column sums) without a date, aircraft type, or airport codes. They may be labeled "Totals", "Page Total", "Total this page", "Amounts forwarded", "Brought forward", etc., or they may just be an unlabeled row of numbers at the bottom of the page. These are NOT flights.
- The logbook is in standard ASA/Jeppesen format
- Include ALL columns you can read - do not skip any time categories

Return a JSON object with two fields:
1. "total_rows_visible": the total number of flight data rows you can see in the logbook (count every row that has ANY data written in it)
2. "entries": the JSON array of extracted flight entries

The length of "entries" MUST equal "total_rows_visible". If they don't match, you missed rows.

Example:
{"total_rows_visible": 1, "entries": [{"date": "05/09/2004", "aircraft_model": "DA20", "aircraft_ident": "N636DC", "route_from": "BFI", "route_to": "BFI", "remarks": "Stalls, slow flight", "total_duration": 1.4, "pic": 0, "sic": 0, "dual_recd": 1.4, "dual_given": 0, "solo": 0, "cross_country": 0, "night": 0, "actual_inst": 0, "simulated_inst": 0, "num_inst_app": 0, "landings_day": 3, "landings_night": 0}]}"""

    # Gemini REST API base URL (Generative Language API)
    _GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    # Model name — env-overridable so a future Google model change is config, not code.
    _GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

    def __init__(self):
        """Initialize OCR service."""
        self._client = None          # google-genai Client (API key path)
        self._sa_credentials = None  # service account credentials (REST path)
        self._initialized = False
        self._use_sdk = False        # True = google-genai SDK, False = REST API

    def _init_gemini(self):
        """Initialize Gemini: prefer API key + SDK, fall back to service account + REST."""
        if self._initialized:
            return

        try:
            # Option 1: API key via google-genai SDK (simplest)
            api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            if api_key and GENAI_SDK_AVAILABLE:
                self._client = genai.Client(api_key=api_key)
                self._use_sdk = True
                self._initialized = True
                print("Gemini initialized with API key (SDK)")
                return

            if api_key:
                # API key but no SDK — use REST with key param
                self._api_key = api_key
                self._use_sdk = False
                self._initialized = True
                print("Gemini initialized with API key (REST)")
                return

            # Option 2: Service account → REST API (Generative Language API)
            sa_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON', '')
            sa_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

            # Check project root and /tmp for existing service-account.json
            if not sa_path:
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                for candidate in [
                    os.path.join(project_root, 'service-account.json'),
                    '/tmp/service-account.json',
                ]:
                    if os.path.exists(candidate):
                        sa_path = candidate
                        break

            # Create from env var if no file found yet
            if not sa_path or not os.path.exists(sa_path):
                if sa_json:
                    sa_path = '/tmp/service-account.json'
                    with open(sa_path, 'w') as f:
                        f.write(sa_json)
                    print(f"Created {sa_path} from GOOGLE_SERVICE_ACCOUNT_JSON env var ({len(sa_json)} chars)")
                else:
                    print("Error: No GEMINI_API_KEY, no service-account.json, and "
                          "GOOGLE_SERVICE_ACCOUNT_JSON env var is empty.")
                    return

            if os.path.exists(sa_path):
                from google.oauth2.service_account import Credentials
                from google.auth.transport.requests import Request
                self._sa_credentials = Credentials.from_service_account_file(
                    sa_path,
                    scopes=['https://www.googleapis.com/auth/generative-language'],
                )
                self._sa_credentials.refresh(Request())
                self._use_sdk = False
                self._api_key = None
                self._initialized = True
                print(f"Gemini initialized with service account (REST) from {sa_path}")
                return

            print("Error: No GEMINI_API_KEY and no service-account.json found.")

        except Exception as e:
            print(f"Error initializing Gemini: {e}")

    _MAX_OUTPUT_TOKENS = 32768   # busy pages truncated at 8192 -> malformed JSON

    def _gemini_generate(self, prompt: str, image_bytes: bytes) -> str:
        """Call Gemini via SDK or REST. Retries on rate-limit (429) / transient
        (500/503/timeout) with exponential backoff so rapid-fire pages pace out."""
        import base64, time

        attempts = 6
        for attempt in range(attempts):
            try:
                if self._use_sdk and self._client:
                    image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                    response = self._client.models.generate_content(
                        model=self._GEMINI_MODEL,
                        contents=[prompt, image_part],
                        config=genai_types.GenerateContentConfig(
                            temperature=0.1, max_output_tokens=self._MAX_OUTPUT_TOKENS),
                    )
                    return response.text

                # REST API path (service account or API key without SDK)
                url = f"{self._GEMINI_API_URL}/{self._GEMINI_MODEL}:generateContent"
                headers = {"Content-Type": "application/json"}
                params = {}
                if self._sa_credentials:
                    from google.auth.transport.requests import Request
                    if not self._sa_credentials.valid:
                        self._sa_credentials.refresh(Request())
                    headers["Authorization"] = f"Bearer {self._sa_credentials.token}"
                elif getattr(self, '_api_key', None):
                    params["key"] = self._api_key

                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                body = {
                    "contents": [{"parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": "image/jpeg", "data": image_b64}},
                    ]}],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": self._MAX_OUTPUT_TOKENS},
                }
                resp = _requests.post(url, headers=headers, params=params, json=body, timeout=180)
                if resp.status_code in (429, 500, 503) and attempt < attempts - 1:
                    ra = resp.headers.get("Retry-After")
                    wait = int(ra) if (ra and str(ra).isdigit()) else min(64, 6 * (2 ** attempt))
                    print(f"Gemini {resp.status_code}; retry {attempt+1}/{attempts} in {wait}s")
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                msg = str(e)
                transient = any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "500", "503", "overloaded", "timed out", "timeout"))
                if transient and attempt < attempts - 1:
                    wait = min(64, 6 * (2 ** attempt))
                    print(f"Gemini transient ({msg[:60]}); retry {attempt+1}/{attempts} in {wait}s")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("Gemini generate failed after retries")

    def extract_flights_with_gemini(self, image_path: str,
                                     known_idents: set[str] | None = None,
                                     known_models: set[str] | None = None,
                                     known_airports: set[str] | None = None,
                                     ) -> tuple[list[dict], int, int]:
        """
        Extract flight entries from logbook image using Google Gemini.

        Args:
            image_path: Path to logbook image
            known_idents: Known aircraft tail numbers from existing entries
            known_models: Known aircraft types from existing entries
            known_airports: Known airport codes from existing entries

        Returns:
            Tuple of (entries, expected_rows, actual_rows)
        """
        self._init_gemini()

        if not self._initialized:
            print("ERROR: Gemini not initialized — set GEMINI_API_KEY or provide service-account.json")
            return [], 0, 0

        try:
            # Load image
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            print(f"Sending {len(image_bytes)} byte image to Gemini...")

            raw_text = self._gemini_generate(self.EXTRACTION_PROMPT, image_bytes)

            # Parse response
            response_text = raw_text.strip()
            print(f"Gemini response length: {len(response_text)} chars")

            # Clean up response - remove markdown code blocks if present
            if response_text.startswith("```"):
                # Remove ```json and ``` markers
                lines = response_text.split('\n')
                lines = [l for l in lines if not l.strip().startswith('```')]
                response_text = '\n'.join(lines)

            # Parse JSON
            parsed = json.loads(response_text)

            # Handle both formats: object with total_rows_visible or plain list
            if isinstance(parsed, dict):
                expected_rows = parsed.get('total_rows_visible', 0)
                entries = parsed.get('entries', [])
                print(f"Gemini reports {expected_rows} rows visible, extracted {len(entries)} entries")

                if expected_rows > len(entries):
                    print(f"WARNING: Missing {expected_rows - len(entries)} rows! "
                          f"Expected {expected_rows}, got {len(entries)}")
            elif isinstance(parsed, list):
                entries = parsed
                expected_rows = len(entries)
            else:
                print(f"WARNING: Gemini returned unexpected type: {type(parsed)}")
                return [], 0, 0

            print(f"Gemini extracted {len(entries)} flight entries")

            # Normalize entries
            normalized = []
            for entry in entries:
                normalized.append(self._normalize_entry(entry))

            # Filter out summary/totals rows that Gemini missed
            filtered = [e for e in normalized if self._is_flight_entry(e)]
            if len(filtered) < len(normalized):
                print(f"Filtered out {len(normalized) - len(filtered)} summary/totals row(s)")

            # Fix dates: normalize format, propagate years, handle Dec→Jan rollover
            filtered = self._normalize_dates(filtered)

            # Fix aircraft identifiers using frequency analysis + known idents
            filtered = self._fix_aircraft_idents(filtered, known_idents)

            # Fix aircraft types against known FAA designators (multi-stage pipeline)
            filtered = self._fix_aircraft_models(filtered, known_models)

            # Fix airport codes using frequency analysis + known airports
            filtered = self._fix_airport_codes(filtered, known_airports)

            # Normalize airport codes to ICAO format (BLI→KBLI, SUN→KSUN)
            filtered = self._normalize_codes_to_icao(filtered)

            # Ensure consistent type per tail number registration
            filtered = self._fix_type_per_registration(filtered)

            # Migrate FTD time: move total_duration → sim for FTD entries
            filtered = self._migrate_ftd_time(filtered)

            # Extract multi-leg routes from remarks into route_via
            filtered = self._extract_multi_leg_routes(filtered)

            # Fix airport codes in route_via (after multi-leg extraction)
            filtered = self._fix_airport_code_in_route_via(filtered)

            # Clean up garbled remarks
            filtered = self._clean_remarks(filtered)

            # Fix landings for single-leg flights (common OCR misread: 1→2)
            filtered = self._fix_landings(filtered)

            return filtered, expected_rows, len(filtered)

        except json.JSONDecodeError as e:
            print(f"Error parsing Gemini JSON response: {e}")
            print(f"Response was: {response_text[:500]}")
            return [], 0, 0
        except Exception as e:
            print(f"Error with Gemini extraction: {e}")
            return [], 0, 0

    def _normalize_dates(self, entries: list[dict]) -> list[dict]:
        """Fix dates across all entries: normalize format, propagate years, handle Dec→Jan rollover."""
        # First pass: normalize each date individually, preserve date ranges in remarks
        for entry in entries:
            raw_date = entry.get('date', '')
            normalized, date_range = self._normalize_date(raw_date)
            entry['date'] = normalized
            if date_range:
                remarks = entry.get('remarks', '').strip()
                entry['remarks'] = f"{date_range} {remarks}".strip() if remarks else date_range
                print(f"Date range '{raw_date}' → date='{normalized}', added range to remarks")

        # Second pass: propagate years to entries missing them
        # Find the most common year from entries that have one
        years = []
        for entry in entries:
            m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})$', entry['date'])
            if m:
                years.append(m.group(3))

        default_year = max(set(years), key=years.count) if years else None

        for entry in entries:
            date = entry['date']
            # If date is just MM/DD with no year, add the default year
            m = re.match(r'^(\d{1,2})/(\d{1,2})$', date)
            if m and default_year:
                entry['date'] = f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{default_year}"
                print(f"Date fix: '{date}' → '{entry['date']}' (added year)")

        # Third pass: handle Dec→Jan rollover within the page
        # If dates go from late months (Oct-Dec) to early months (Jan-Mar),
        # bump the year by 1 for entries after the rollover point.
        self._fix_year_rollover(entries)

        return entries

    @staticmethod
    def _fix_year_rollover(entries: list[dict]) -> None:
        """Detect and fix a single Dec→Jan year rollover within a page.

        If dates go Oct-Dec then Jan-Mar, bump the year +1 for the Jan+ entries.
        Only allows one rollover per page (conservative).
        """
        # Extract months from entries
        entry_months = []
        for e in entries:
            m = re.match(r'(\d{1,2})/\d{1,2}/\d{4}', e.get('date', ''))
            entry_months.append(int(m.group(1)) if m else None)

        # Find a single Dec→Jan rollover point
        valid = [(i, m) for i, m in enumerate(entry_months) if m is not None]
        rollover_idx = None

        for j in range(len(valid) - 1):
            idx_a, month_a = valid[j]
            idx_b, month_b = valid[j + 1]
            if month_a >= 10 and month_b <= 3:
                if rollover_idx is None:
                    rollover_idx = idx_b
                else:
                    rollover_idx = None  # Multiple rollovers → don't do any
                    break

        if rollover_idx is None:
            return

        # Get the base year from entries before rollover
        base_year = None
        for i in range(rollover_idx):
            m = re.match(r'\d{1,2}/\d{1,2}/(\d{4})', entries[i].get('date', ''))
            if m:
                base_year = int(m.group(1))

        if base_year is None:
            return

        # Bump year for entries at and after rollover
        for idx in range(rollover_idx, len(entries)):
            m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', entries[idx].get('date', ''))
            if not m:
                continue
            month, day, old_year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if old_year == base_year:
                entries[idx]['date'] = f"{month:02d}/{day:02d}/{base_year + 1}"
                print(f"Rollover fix: {m.group(0)} → {entries[idx]['date']}")

    def _normalize_date(self, date_str: str) -> tuple[str, str | None]:
        """Normalize a single date string to MM/DD/YYYY format with leading zeros.

        Returns:
            (normalized_date, date_range_or_None) — date_range is the original
            range string (e.g. "3/26-3/28/2023") if the date was a range, else None.
        """
        date_str = date_str.strip()
        if not date_str:
            return '', None

        date_range = None

        # Handle date ranges like "3/26-3/28/2023" — use first date, save range
        range_match = re.match(r'^(\d{1,2})/(\d{1,2})\s*-\s*(\d{1,2})/(\d{1,2})/(\d{1,4})$', date_str)
        if range_match:
            date_range = date_str
            month, day = range_match.group(1), range_match.group(2)
            year = range_match.group(5)
            date_str = f"{month}/{day}/{year}"
        else:
            # "3/26-28/2023" format (same month)
            range_match2 = re.match(r'^(\d{1,2})/(\d{1,2})\s*-\s*(\d{1,2})/(\d{1,4})$', date_str)
            if range_match2:
                date_range = date_str
                month, day = range_match2.group(1), range_match2.group(2)
                year = range_match2.group(4)
                date_str = f"{month}/{day}/{year}"

        # Now parse M/D/YYYY or M/D
        m = re.match(r'^(\d{1,2})/(\d{1,2})(?:/(\d{1,4}))?$', date_str)
        if not m:
            return date_str, date_range  # Can't parse, return as-is

        month, day, year = int(m.group(1)), int(m.group(2)), m.group(3)

        if year:
            year = self._fix_year(year)
            return f"{month:02d}/{day:02d}/{year}", date_range
        else:
            # No year — return MM/DD, will be fixed in second pass
            return f"{month:02d}/{day:02d}", date_range

    def _fix_year(self, year_str: str) -> str:
        """Fix garbled year strings to valid 4-digit years."""
        year_str = year_str.strip()
        n = int(year_str)

        if 1950 <= n <= 2099:
            return str(n)

        # 1-2 digit: assume 2000s (e.g., "4" → "2004", "23" → "2023")
        if n <= 99:
            return str(2000 + n) if n <= 50 else str(1900 + n)

        # 3-digit garbled year: e.g., "206" → "2006", "202" → "2020"?
        # Most likely a dropped digit from a 200x year
        s = year_str
        if len(s) == 3:
            # Try inserting a '0' at each position to make a valid 4-digit year
            for i in range(len(s) + 1):
                candidate = s[:i] + '0' + s[i:]
                cn = int(candidate)
                if 1990 <= cn <= 2099:
                    print(f"Year fix: '{year_str}' → '{candidate}'")
                    return candidate
            # Fallback: prepend '2' if starts with '0'
            if s.startswith('0'):
                return '2' + s

        return year_str  # Can't fix, return as-is

    def _is_flight_entry(self, entry: dict) -> bool:
        """Return True if this looks like an actual flight, not a totals/summary row."""
        date_str = entry.get('date', '').strip()
        # Date must contain digits and a separator (/ or -) to be real
        has_date = bool(date_str) and bool(re.search(r'\d+[/\-]\d+', date_str))
        has_aircraft = bool(entry.get('aircraft_model', '').strip()) or bool(entry.get('aircraft_ident', '').strip())
        has_route = bool(entry.get('route_from', '').strip()) or bool(entry.get('route_to', '').strip())
        has_duration = float(entry.get('total_duration', 0) or 0) > 0
        # Must have a route or duration to be a real flight (not just carried-forward aircraft info)
        if not has_route and not has_duration:
            return False
        # Require at least 2 of 3 indicators to be a real flight
        score = sum([has_date, has_aircraft, has_route])
        return score >= 2

    # Common OCR character confusions (bidirectional)
    OCR_CONFUSIONS = {
        'Y': '7', '7': 'Y',
        'T': '7',
        'O': '0', '0': 'O',
        'I': '1', '1': 'I',
        'S': '5', '5': 'S',
        'B': '8', '8': 'B',
        'G': '6', '6': 'G',
        'Z': '2', '2': 'Z',
        'D': '0',
    }

    # Airport database cache: {code: (lat, lon)} for all valid airport codes
    _airport_db: dict[str, tuple[float, float]] | None = None

    @classmethod
    def _load_airport_db(cls) -> dict[str, tuple[float, float]]:
        """Load airport database from CSV. Returns {code: (lat, lon)} for all valid codes.

        Includes both FAA identifiers (BFI) and ICAO identifiers (KBFI).
        When codes conflict between US and international airports, US airports win
        since this logbook is for US-based flying.
        """
        if cls._airport_db is not None:
            return cls._airport_db

        db: dict[str, tuple[float, float]] = {}
        db_is_us: dict[str, bool] = {}  # Track country for conflict resolution
        csv_path = Path(__file__).parent.parent / 'data' / 'airports.csv'
        if not csv_path.exists():
            print(f"Warning: Airport database not found at {csv_path}")
            cls._airport_db = db
            return db

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lat_str = row.get('latitude_deg', '')
                lon_str = row.get('longitude_deg', '')
                if not lat_str or not lon_str:
                    continue
                try:
                    lat, lon = float(lat_str), float(lon_str)
                except ValueError:
                    continue
                is_us = row.get('iso_country', '') == 'US'
                # Add all code variants; US airports override international ones
                for field in ('ident', 'local_code', 'gps_code', 'iata_code'):
                    code = (row.get(field) or '').strip().upper()
                    if not code:
                        continue
                    if code not in db:
                        db[code] = (lat, lon)
                        db_is_us[code] = is_us
                    elif is_us and not db_is_us.get(code, False):
                        db[code] = (lat, lon)
                        db_is_us[code] = True

        cls._airport_db = db
        print(f"Loaded {len(db)} airport codes from database")
        return db

    @staticmethod
    def _geo_distance_nm(a: tuple[float, float], b: tuple[float, float]) -> float:
        """Great-circle distance between two (lat, lon) points in nautical miles."""
        lat1, lon1 = math.radians(a[0]), math.radians(a[1])
        lat2, lon2 = math.radians(b[0]), math.radians(b[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * 3440.065 * math.asin(math.sqrt(min(h, 1.0)))

    # Map common full/informal names → FAA type designator
    AIRCRAFT_NAME_MAP = {
        'PIPERCUB': 'J3', 'PIPERJ3': 'J3', 'PIPERJ3': 'J3', 'J3CUB': 'J3', 'J3': 'J3',
        'CESSNA150': 'C150', 'CESSNA152': 'C152', 'CESSNA172': 'C172',
        'CESSNA177': 'C177', 'CESSNA180': 'C180', 'CESSNA182': 'C182',
        'CESSNA185': 'C185', 'CESSNA206': 'C206', 'CESSNA210': 'C210',
        'CESSNA310': 'C310', 'CESSNA340': 'C340',
        'DIAMONDDA20': 'DA20', 'DIAMONDDA40': 'DA40', 'DIAMONDDA42': 'DA42', 'DIAMONDDA62': 'DA62',
        'DIAMOND20': 'DA20', 'DIAMOND40': 'DA40', 'DIAMOND42': 'DA42',
        'BONANZA': 'BE36', 'BARON': 'BE58',
        'CIRRUSSR20': 'SR20', 'CIRRUSSR22': 'SR22',
        'MALIBU': 'PA46', 'SEMINOLE': 'PA44', 'WARRIOR': 'PA28', 'ARCHER': 'PA28',
        'CHEROKEE': 'PA28', 'SENECA': 'PA34', 'ARROW': 'PA28',
        'SKYHAWK': 'C172', 'SKYLANE': 'C182',
    }

    # Common FAA aircraft type designators
    KNOWN_AIRCRAFT_TYPES = {
        # Cessna
        'C150', 'C152', 'C170', 'C172', 'C172S', 'C172R', 'C172N', 'C172P',
        'C175', 'C177', 'C180', 'C182', 'C182T',
        'C185', 'C190', 'C195', 'C206', 'C206H', 'C206T', 'C207', 'C208', 'C210', 'C210T',
        'C310', 'C320', 'C335', 'C337', 'C340', 'C402', 'C404', 'C414', 'C421',
        'C425', 'C441', 'C500', 'C510', 'C525', 'C550', 'C560',
        # Piper
        'J3', 'PA11', 'PA12', 'PA18', 'PA20', 'PA22', 'PA23', 'PA24', 'PA28',
        'PA30', 'PA31', 'PA32', 'PA34', 'PA38', 'PA44', 'PA46', 'PA46T',
        # Beechcraft
        'BE33', 'BE35', 'BE36', 'BE55', 'BE58', 'BE60', 'BE76', 'BE95',
        'BE99', 'BE9L', 'B200', 'B300', 'B350',
        # Mooney
        'M20', 'M20T', 'M20J', 'M20K', 'M20R', 'M20S',
        # Diamond
        'DA20', 'DA40', 'DA42', 'DA62',
        # Cirrus
        'SR20', 'SR22', 'SF50',
        # SOCATA / Daher / TBM
        'TBM7', 'TBM8', 'TBM9', 'TB9', 'TB10', 'TB20', 'TB21',
        # North American / Military trainers
        'T6', 'T34', 'T28',
        # CASA / Airbus Military
        'CH2T',
        # Douglas
        'DC3', 'DC3A', 'DC6',
        # Grumman / American General
        'AA1', 'AA5', 'GA7',
        # Maule
        'M4', 'M5', 'M6', 'M7',
        # Vans RV
        'RV4', 'RV6', 'RV7', 'RV8', 'RV9', 'RV10', 'RV12', 'RV14',
        # Pilatus
        'PC12', 'PC6', 'PC24',
        # Extra / Aerobatic
        'E300', 'E330', 'E530',
        # Boeing / Airbus (airline types)
        'B737', 'B738', 'B739', 'B752', 'B753', 'B763', 'B772', 'B77W',
        'B788', 'B789', 'A319', 'A320', 'A321', 'A332', 'A333', 'A339',
        'A346', 'A359', 'A388',
        # Embraer
        'E170', 'E175', 'E190', 'E195', 'E75L', 'E75S',
        # Bombardier / CRJ
        'CRJ2', 'CRJ7', 'CRJ9', 'CL30', 'CL35', 'CL60', 'GL5T', 'GLEX',
        # De Havilland / Viking
        'DHC2', 'DHC3', 'DHC6',
        # Robinson Helicopters
        'R22', 'R44', 'R66',
        # Bell Helicopters
        'B206', 'B407', 'B412',
        # Eurocopter / Airbus Helicopters
        'AS50', 'EC30', 'EC35', 'EC45',
    }

    def _fix_aircraft_idents(self, entries: list[dict], known_idents: set[str] | None = None) -> list[dict]:
        """Fix aircraft identifiers using frequency analysis and OCR character correction.

        If a tail number appears rarely but is 1 character off from a frequent one,
        correct it to the frequent version (e.g., N61637 → N6163Y).
        Also uses known_idents from existing database entries.
        """
        # Count ident frequencies across extracted entries
        ident_counts = Counter(e['aircraft_ident'] for e in entries if e.get('aircraft_ident'))

        # Merge with known idents (treat them as high-frequency)
        if known_idents:
            for ident in known_idents:
                ident_counts[ident] = ident_counts.get(ident, 0) + 10  # Boost known idents

        # Build correction map for rare idents
        corrections = {}
        for ident, count in list(ident_counts.items()):
            if count > 2:
                continue  # Likely correct already
            # Check if there's a frequent ident that's 1 OCR-char away
            best_match = self._find_ocr_match(ident, ident_counts, min_freq=2)
            if best_match:
                corrections[ident] = best_match

        # Apply corrections
        for entry in entries:
            ident = entry.get('aircraft_ident', '')
            if ident in corrections:
                print(f"Ident fix: '{ident}' → '{corrections[ident]}'")
                entry['aircraft_ident'] = corrections[ident]

        return entries

    def _find_ocr_match(self, value: str, candidates: Counter, min_freq: int = 2) -> str | None:
        """Find a frequently-occurring candidate that differs by 1 OCR-confusable character."""
        if not value:
            return None
        for candidate, freq in candidates.items():
            if freq < min_freq or candidate == value:
                continue
            if len(candidate) != len(value):
                continue
            # Count character differences
            diffs = []
            for i, (a, b) in enumerate(zip(value.upper(), candidate.upper())):
                if a != b:
                    diffs.append((i, a, b))
            if len(diffs) != 1:
                continue
            # Check if the single diff is an OCR confusion
            _, char_a, char_b = diffs[0]
            if self.OCR_CONFUSIONS.get(char_a) == char_b or self.OCR_CONFUSIONS.get(char_b) == char_a:
                return candidate
        return None

    def _find_general_frequency_match(self, value: str, candidates: Counter, min_ratio: int = 5) -> str | None:
        """Find a much-more-frequent candidate that differs by exactly 1 character.

        Unlike _find_ocr_match, this doesn't require the character diff to be in
        OCR_CONFUSIONS. Used for cases like DH40→DA40 where H→A isn't a standard
        OCR confusion but the frequency strongly suggests a misread.
        """
        if not value:
            return None
        my_freq = candidates.get(value, 0)
        for candidate, freq in candidates.items():
            if candidate == value or len(candidate) != len(value):
                continue
            # Require the candidate to be much more frequent
            if freq < max(my_freq * min_ratio, min_ratio):
                continue
            # Count character differences - must be exactly 1
            diffs = sum(1 for a, b in zip(value, candidate) if a != b)
            if diffs == 1:
                return candidate
        return None

    def _detect_ftd(self, raw_model: str) -> tuple[str | None, str | None]:
        """Detect FTD/simulator entries and return (normalized_type, base_type).

        Returns (None, None) if not an FTD.
        Examples:
            'FTD42' → ('DA42-FTD', 'DA42')
            'DA42FTD' → ('DA42-FTD', 'DA42')
            'DA42SM' → ('DA42-FTD', 'DA42')
            'PTO42' → ('DA42-FTD', 'DA42')
            'Frasca 142' → ('Frasca-FTD', None)
            'FRESA192' → ('Frasca-FTD', None)
            'FRESA100' → ('Frasca-FTD', None)
            'Frusia 142' → ('Frasca-FTD', None)
        """
        upper = raw_model.upper().strip()
        normed = re.sub(r'[-\s]', '', upper)

        # Already has -FTD suffix (e.g., DA42-FTD)
        m = re.match(r'^([A-Z]+\d+)-?FTD$', normed)
        if m:
            base = m.group(1)
            return f"{base}-FTD", base

        # TYPE-SM pattern (e.g., DA42SM → simulator)
        m = re.match(r'^([A-Z]+\d+)SM$', normed)
        if m:
            base = m.group(1)
            return f"{base}-FTD", base

        # FTD + digits pattern (e.g., FTD42 → try DA42)
        m = re.match(r'^FTD(\d+)$', normed)
        if m:
            digits = m.group(1)
            # Try common prefixes: DA, C, PA
            for prefix in ['DA', 'C', 'PA']:
                candidate = prefix + digits
                if candidate in self.KNOWN_AIRCRAFT_TYPES:
                    return f"{candidate}-FTD", candidate
            return f"FTD{digits}-FTD", None

        # PTO + digits pattern (e.g., PTO42 → try DA42)
        m = re.match(r'^PTO(\d+)$', normed)
        if m:
            digits = m.group(1)
            for prefix in ['DA', 'C', 'PA']:
                candidate = prefix + digits
                if candidate in self.KNOWN_AIRCRAFT_TYPES:
                    return f"{candidate}-FTD", candidate
            return f"FTD{digits}-FTD", None

        # Frasca/Frusia/Fresa patterns (various OCR misreadings)
        if re.match(r'^(FRASCA|FRUSIA|FRESA|FRESCA)', normed):
            return 'Frasca-FTD', None

        return None, None

    def _fix_aircraft_models(self, entries: list[dict], known_models: set[str] | None = None) -> list[dict]:
        """Fix aircraft type designators using multi-stage pipeline.

        Pipeline order (stops at first match):
        1. Full name mapping (Piper Cub → J3)
        2. FTD/simulator detection
        3. Known type check
        4. Variant suffix stripping (DA20C1 → DA20)
        5. OCR prefix fixes (K182T → C182T)
        6. OCR single-char substitution
        7. Frequency-based correction within batch
        8. Garbage detection (pure numeric, tail-number-like)
        """
        valid_types = set(self.KNOWN_AIRCRAFT_TYPES)
        if known_models:
            valid_types.update(known_models)

        # Normalize: strip hyphens/spaces for comparison
        def normalize_model(m):
            return re.sub(r'[-\s]', '', m).upper()

        # Build a lookup of normalized known types
        normalized_known = {}
        for t in valid_types:
            normalized_known[normalize_model(t)] = t

        # Count model frequencies across entries
        model_counts = Counter(normalize_model(e['aircraft_model'])
                               for e in entries if e.get('aircraft_model'))

        for entry in entries:
            raw = entry.get('aircraft_model', '').strip()
            if not raw:
                continue
            normed = normalize_model(raw)

            # Stage 1: Full name mapping
            name_key = re.sub(r'[-\s]', '', raw.upper())
            if name_key in self.AIRCRAFT_NAME_MAP:
                result = self.AIRCRAFT_NAME_MAP[name_key]
                if result != raw:
                    print(f"Model fix (name): '{raw}' → '{result}'")
                entry['aircraft_model'] = result
                continue

            # Stage 2: FTD/simulator detection
            ftd_type, base_type = self._detect_ftd(raw)
            if ftd_type:
                print(f"Model fix (FTD): '{raw}' → '{ftd_type}'")
                entry['aircraft_model'] = ftd_type
                entry['_is_ftd'] = True
                continue

            # Stage 3: Known type check
            if normed in normalized_known:
                entry['aircraft_model'] = normalized_known[normed]
                continue

            # Stage 4: Variant suffix stripping (DA20C1 → DA20, DA20E1 → DA20)
            # Only strip if the base is known but the full string is NOT known
            m = re.match(r'^([A-Z]+\d+)[A-Z]\d*$', normed)
            if m:
                base = m.group(1)
                if base in normalized_known and normed not in normalized_known:
                    print(f"Model fix (variant): '{raw}' → '{normalized_known[base]}'")
                    entry['aircraft_model'] = normalized_known[base]
                    continue

            # Stage 5: OCR prefix fixes
            # K### → C### (e.g., K182T → C182T)
            if re.match(r'^K\d', normed):
                candidate = 'C' + normed[1:]
                if candidate in normalized_known:
                    print(f"Model fix (prefix K→C): '{raw}' → '{normalized_known[candidate]}'")
                    entry['aircraft_model'] = normalized_known[candidate]
                    continue
            # A## → DA## (e.g., A40 → DA40)
            if re.match(r'^A\d{2}$', normed):
                candidate = 'DA' + normed[1:]
                if candidate in normalized_known:
                    print(f"Model fix (prefix A→DA): '{raw}' → '{normalized_known[candidate]}'")
                    entry['aircraft_model'] = normalized_known[candidate]
                    continue
            # FB## → possible misread
            if re.match(r'^FB\d', normed):
                # Try common fixes: FB75 could be misread
                pass  # No clear mapping, fall through

            # Stage 6: OCR single-char substitution
            fixed = self._try_ocr_fix_against_set(normed, normalized_known)
            if fixed:
                print(f"Model fix (OCR): '{raw}' → '{fixed}'")
                entry['aircraft_model'] = fixed
                continue

            # Stage 7: Frequency-based correction within batch (OCR confusion only)
            freq_match = self._find_ocr_match(normed, model_counts, min_freq=2)
            if freq_match and freq_match in normalized_known:
                print(f"Model fix (freq): '{raw}' → '{normalized_known[freq_match]}'")
                entry['aircraft_model'] = normalized_known[freq_match]
                continue

            # Stage 7b: General frequency correction (any single-char diff, high ratio)
            # Handles cases like DH40→DA40 where the diff isn't in OCR_CONFUSIONS
            freq_match = self._find_general_frequency_match(normed, model_counts, min_ratio=5)
            if freq_match and freq_match in normalized_known:
                print(f"Model fix (freq-general): '{raw}' → '{normalized_known[freq_match]}'")
                entry['aircraft_model'] = normalized_known[freq_match]
                continue

            # Stage 8: Garbage detection
            # Pure numeric (e.g., "844")
            if re.match(r'^\d+$', normed):
                print(f"Model fix (garbage numeric): '{raw}' → cleared")
                entry['aircraft_model'] = ''
                continue
            # Tail-number-like (starts with digits followed by letters, e.g., "556DC")
            if re.match(r'^\d{2,}[A-Z]+$', normed):
                print(f"Model fix (garbage tail-like): '{raw}' → cleared")
                entry['aircraft_model'] = ''
                continue
            # Single char unknown — likely garbage
            if len(normed) <= 1:
                print(f"Model fix (garbage short): '{raw}' → cleared")
                entry['aircraft_model'] = ''
                continue

            # If nothing matched, leave as-is (might be a valid but unlisted type)
            print(f"Model unknown (keeping): '{raw}'")

        return entries

    def _try_ocr_fix_against_set(self, value: str, known_set: dict) -> str | None:
        """Try single-character OCR substitutions to match a known value."""
        for i, char in enumerate(value):
            # Try direct confusion mapping
            alt = self.OCR_CONFUSIONS.get(char)
            if alt:
                candidate = value[:i] + alt + value[i+1:]
                if candidate in known_set:
                    return known_set[candidate]
            # Also try reverse: what if this char is the correct one
            # and another mapping would work
            for orig, replacement in self.OCR_CONFUSIONS.items():
                if replacement == char:
                    candidate = value[:i] + orig + value[i+1:]
                    if candidate in known_set:
                        return known_set[candidate]
        return None

    # Cache of nearby airports per home base: {home_code: {code: (lat, lon)}}
    _nearby_airport_cache: dict[str, dict[str, tuple[float, float]]] = {}

    def _get_nearby_airports(self, home_pos: tuple[float, float], home_code: str,
                              radius_nm: float = 500.0) -> dict[str, tuple[float, float]]:
        """Get all valid airport codes within radius of home base. Cached per home code."""
        if home_code in self._nearby_airport_cache:
            return self._nearby_airport_cache[home_code]

        airport_db = self._load_airport_db()
        nearby = {}
        for code, pos in airport_db.items():
            if self._geo_distance_nm(home_pos, pos) <= radius_nm:
                nearby[code] = pos

        self._nearby_airport_cache[home_code] = nearby
        print(f"Built nearby airport set: {len(nearby)} airports within {radius_nm}nm of {home_code}")
        return nearby

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        m, n = len(a), len(b)
        if m == 0:
            return n
        if n == 0:
            return m
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]

    def _fix_airport_codes(self, entries: list[dict], known_airports: set[str] | None = None) -> list[dict]:
        """Fix airport codes using the FAA airport database and geographic proximity.

        Uses a scoring system combining edit distance, geographic distance from
        home base, and batch frequency to find the best correction.
        """
        airport_db = self._load_airport_db()
        if not airport_db:
            return entries

        # Determine "home base" — most frequent airport in this batch
        airport_counts: Counter = Counter()
        for e in entries:
            for field in ('route_from', 'route_to'):
                code = (e.get(field) or '').strip().upper()
                if code:
                    airport_counts[code] += 1

        if not airport_counts:
            return entries

        home_code = airport_counts.most_common(1)[0][0]
        home_pos = airport_db.get(home_code) or airport_db.get('K' + home_code) or airport_db.get(home_code.lstrip('K'))
        if not home_pos:
            print(f"Warning: home base '{home_code}' not found in airport database")
            return entries

        print(f"Airport fix: home base = {home_code} ({home_pos[0]:.2f}, {home_pos[1]:.2f})")

        nearby = self._get_nearby_airports(home_pos, home_code)

        # Build correction map
        corrections: dict[str, str] = {}

        all_codes_in_entries = set()
        for e in entries:
            for field in ('route_from', 'route_to'):
                code = (e.get(field) or '').strip().upper()
                if code:
                    all_codes_in_entries.add(code)

        for code in all_codes_in_entries:
            fixed = self._fix_single_airport_code(code, airport_db, home_pos, nearby, airport_counts)
            if fixed and fixed != code:
                corrections[code] = fixed

        # Apply corrections to route_from/route_to
        for entry in entries:
            for field in ('route_from', 'route_to'):
                code = (entry.get(field) or '').strip().upper()
                if code in corrections:
                    print(f"Airport fix: '{code}' → '{corrections[code]}' ({field})")
                    entry[field] = corrections[code]

        return entries

    def _normalize_codes_to_icao(self, entries: list[dict]) -> list[dict]:
        """Normalize all airport codes to ICAO format (BLI→KBLI, SUN→KSUN)."""
        from services.airport_lookup import to_icao

        for entry in entries:
            for field in ('route_from', 'route_to'):
                code = (entry.get(field) or '').strip()
                if code:
                    icao = to_icao(code)
                    if icao != code:
                        entry[field] = icao

            # Also normalize route_via legs
            route_via = (entry.get('route_via') or '').strip()
            if route_via:
                legs = [to_icao(leg.strip()) for leg in route_via.split('-') if leg.strip()]
                entry['route_via'] = '-'.join(legs)

        return entries

    def _fix_airport_code_in_route_via(self, entries: list[dict]) -> list[dict]:
        """Fix airport codes within route_via strings using the airport database.

        Called after _extract_multi_leg_routes to fix codes in multi-leg routes.
        """
        airport_db = self._load_airport_db()
        if not airport_db:
            return entries

        # Determine home base from route_from/route_to (already fixed)
        airport_counts: Counter = Counter()
        for e in entries:
            for field in ('route_from', 'route_to'):
                code = (e.get(field) or '').strip().upper()
                if code:
                    airport_counts[code] += 1
            # Also count codes in route_via for frequency
            for leg in (e.get('route_via') or '').split('-'):
                leg = leg.strip().upper()
                if leg:
                    airport_counts[leg] += 1

        if not airport_counts:
            return entries

        home_code = airport_counts.most_common(1)[0][0]
        home_pos = airport_db.get(home_code) or airport_db.get('K' + home_code) or airport_db.get(home_code.lstrip('K'))
        if not home_pos:
            return entries

        nearby = self._get_nearby_airports(home_pos, home_code)

        for entry in entries:
            route_via = (entry.get('route_via') or '').strip()
            if not route_via:
                continue

            legs = route_via.split('-')
            fixed_legs = []
            changed = False
            for leg in legs:
                leg = leg.strip().upper()
                fixed = self._fix_single_airport_code(leg, airport_db, home_pos, nearby, airport_counts)
                if fixed and fixed != leg:
                    print(f"Route via fix: '{leg}' → '{fixed}'")
                    changed = True
                    fixed_legs.append(fixed)
                else:
                    fixed_legs.append(leg)

            if changed:
                entry['route_via'] = '-'.join(fixed_legs)
                entry['route_from'] = fixed_legs[0]
                entry['route_to'] = fixed_legs[-1]

        return entries

    @staticmethod
    def _strip_k_prefix(code: str) -> str:
        """Strip ICAO K-prefix from US airport codes (KBFI→BFI).

        Only strips K from 4-char codes starting with K, since the K-prefix convention
        only applies to 3-letter FAA codes getting a K prefix for ICAO (BFI→KBFI).
        2-char codes like TY should NOT have K added (KTY is a different airport).
        """
        if len(code) == 4 and code.startswith('K'):
            return code[1:]
        return code

    def _lookup_airport_pos(self, code: str, airport_db: dict) -> tuple[float, float] | None:
        """Look up airport position, trying base code and K-prefix variants."""
        code = code.upper()
        base = self._strip_k_prefix(code)
        pos = airport_db.get(base)
        if not pos and len(base) == 3:
            pos = airport_db.get('K' + base)
        if not pos and base != code:
            pos = airport_db.get(code)
        return pos

    def _get_code_freq(self, code: str, batch_freq: Counter) -> int:
        """Get frequency of an airport code in the batch, combining code forms."""
        base = self._strip_k_prefix(code.upper())
        freq = batch_freq.get(base, 0)
        if len(base) == 3:
            freq = max(freq, batch_freq.get('K' + base, 0))
        return freq

    def _fix_single_airport_code(self, code: str, airport_db: dict,
                                  home_pos: tuple[float, float],
                                  nearby: dict[str, tuple[float, float]],
                                  batch_freq: Counter) -> str | None:
        """Fix a single airport code using the airport database and geographic context.

        Strategy for VALID codes:
        1. Within 100nm of home → always keep (local training destinations)
        2. >3000nm from home → auto-correct to best 1-edit match within 100nm of home
           (clearly international/unreasonable: TIN=Algeria, PHE=Australia, TOY=Japan)
        3. 100-3000nm (plausible cross-country) → correct only if a 1-edit alternative
           within 100nm has higher batch frequency (freq >= 2 AND freq > orig_freq).
           Protects real cross-country destinations (SBA, OSH, BIS) since their 1-edit
           neighbors won't have higher frequency.

        Strategy for INVALID codes:
        4. Fuzzy match against nearby airports (500nm), edit distance ≤ 2,
           scored by edit_distance * 1000 + geo_distance - frequency * 5.
           Edit distance dominates so 1-edit matches always beat 2-edit.
        """
        if not code or len(code) > 5:
            return None

        code = code.upper()
        base_code = self._strip_k_prefix(code)

        code_pos = self._lookup_airport_pos(code, airport_db)
        code_valid = code_pos is not None

        if code_valid:
            code_dist = self._geo_distance_nm(home_pos, code_pos)

            if code_dist <= 100:
                return None  # Close to home, definitely keep

            # Find best 1-edit alternative within 100nm of home
            orig_freq = self._get_code_freq(base_code, batch_freq)

            best = None
            best_freq = 0
            best_dist = float('inf')
            for cand_code, cand_pos in nearby.items():
                cand_base = self._strip_k_prefix(cand_code)
                if self._levenshtein(base_code, cand_base) != 1:
                    continue
                geo_dist = self._geo_distance_nm(home_pos, cand_pos)
                if geo_dist > 100:
                    continue
                cand_freq = self._get_code_freq(cand_base, batch_freq)
                if cand_freq > best_freq or (cand_freq == best_freq and geo_dist < best_dist):
                    best = cand_base
                    best_freq = cand_freq
                    best_dist = geo_dist

            if code_dist > 3000 and best:
                # Unreasonable international distance — auto-correct
                return best

            if code_dist > 200 and best and best_freq >= 2 and best_freq > orig_freq:
                # Plausible distance but frequency strongly favors nearby alternative
                return best

            return None  # Keep original valid code

        # Code is INVALID — find best match among nearby airports
        best_candidate = None
        best_score = float('inf')

        for cand_code, cand_pos in nearby.items():
            cand_base = self._strip_k_prefix(cand_code)

            edit_dist = self._levenshtein(base_code, cand_base)
            if edit_dist > 2 or edit_dist == 0:
                continue

            geo_dist = self._geo_distance_nm(home_pos, cand_pos)
            freq = self._get_code_freq(cand_base, batch_freq)

            # Score: lower is better. Edit distance dominates so 1-edit always beats 2-edit.
            score = edit_dist * 1000 + geo_dist - freq * 5

            if score < best_score:
                best_score = score
                best_candidate = cand_base

        return best_candidate

    def _fix_type_per_registration(self, entries: list[dict]) -> list[dict]:
        """Ensure consistent aircraft type per registration (tail number).

        For each ident with multiple types, pick the most common one (>50% of entries).
        """
        # Build {ident: Counter({model: count})} map
        ident_models: dict[str, Counter] = {}
        for entry in entries:
            ident = entry.get('aircraft_ident', '').strip()
            model = entry.get('aircraft_model', '').strip()
            if ident and model:
                if ident not in ident_models:
                    ident_models[ident] = Counter()
                ident_models[ident][model] += 1

        # For idents with multiple types, correct to the dominant one
        for ident, model_counts in ident_models.items():
            if len(model_counts) <= 1:
                continue
            total = sum(model_counts.values())
            dominant_model, dominant_count = model_counts.most_common(1)[0]
            if dominant_count > total / 2:
                for entry in entries:
                    if entry.get('aircraft_ident') == ident and entry.get('aircraft_model') != dominant_model:
                        old = entry['aircraft_model']
                        # Don't override FTD types
                        if old.endswith('-FTD') or dominant_model.endswith('-FTD'):
                            continue
                        print(f"Type consistency: '{ident}' type '{old}' → '{dominant_model}'")
                        entry['aircraft_model'] = dominant_model

        return entries

    def _migrate_ftd_time(self, entries: list[dict]) -> list[dict]:
        """For FTD entries, move total_duration to sim field."""
        for entry in entries:
            is_ftd = entry.pop('_is_ftd', False)
            model = entry.get('aircraft_model', '')
            if is_ftd or model.endswith('-FTD'):
                if entry.get('total_duration', 0) > 0 and entry.get('sim', 0) == 0:
                    entry['sim'] = entry['total_duration']
                    entry['total_duration'] = 0
                    print(f"FTD time migration: moved {entry['sim']}h to sim for {model}")

        return entries

    # Regex for multi-leg route at start of remarks: airport codes separated by hyphens
    # Matches patterns like "BFI-PAE-BFI", "BFI-TIW-SSO-BFI", "KBFI-KEAT-KBFI"
    _MULTI_LEG_ROUTE_RE = re.compile(
        r'^(K?[A-Z][A-Z0-9]{1,3}(?:\s*-\s*K?[A-Z][A-Z0-9]{1,3}){2,})\s*'
    )

    def _extract_multi_leg_routes(self, entries: list[dict]) -> list[dict]:
        """Extract multi-leg routes from the beginning of remarks into route_via.

        When Gemini puts a route like "BFI-PAE-BFI" at the start of remarks,
        extract it into route_via and remove it from remarks. Also set route_from
        and route_to to the first and last airports.
        """
        for entry in entries:
            remarks = entry.get('remarks', '').strip()
            if not remarks:
                continue

            m = self._MULTI_LEG_ROUTE_RE.match(remarks.upper())
            if not m:
                continue

            route_str = m.group(1).strip()
            # Normalize: remove spaces around hyphens
            route_str = re.sub(r'\s*-\s*', '-', route_str)
            legs = route_str.split('-')

            if len(legs) >= 3:
                entry['route_via'] = route_str
                entry['route_from'] = legs[0]
                entry['route_to'] = legs[-1]
                # Remove the route prefix from remarks
                remaining = remarks[m.end():].strip()
                entry['remarks'] = remaining
                print(f"Multi-leg route extracted: {route_str}")

        return entries

    # Known approach type keywords and common OCR misreads
    _APPROACH_TYPES = {'ILS', 'RNAV', 'VOR', 'LOC', 'NDB', 'GPS', 'LDA', 'SDF', 'VISUAL'}

    # Regex to match approach notation in remarks (OCR output)
    # Captures approach-like patterns including OCR typos (e.g., RNAY for RNAV)
    _APPROACH_RE = re.compile(
        r'\b([A-Z]{2,6})\s*'
        r'(\d{1,2}[LCRZ0-9]*)'
        r'(?:\s+([A-Z]{3,4}))?'
        r'\.?',  # optional trailing period
        re.IGNORECASE
    )

    def _fix_approach_type(self, raw_type: str) -> str | None:
        """Fix OCR typos in approach type (RNAY→RNAV, IIS→ILS, etc.).

        Returns the corrected type or None if not an approach type.
        """
        raw = raw_type.upper()
        if raw in self._APPROACH_TYPES:
            return raw

        # Try 1-edit-distance matches (handles RNAY→RNAV, IIS→ILS, etc.)
        for known in self._APPROACH_TYPES:
            if len(raw) == len(known) and sum(a != b for a, b in zip(raw, known)) == 1:
                return known

        return None

    def _normalize_approach_in_remarks(self, remarks: str, entry: dict | None = None) -> str:
        """Replace OCR approach notation with official FAA approach names.

        E.g. "ILS16RZ PAE" → "ILS RWY 16R KPAE"
             "RNAV 03 SEZ" → "RNAV (GPS) RWY 03 KSEZ"
             "RNAY 34L PAE" → "RNAV (GPS) RWY 34L KPAE" (OCR typo fix)

        Also updates num_inst_app on the entry dict if provided.
        """
        from services.airport_approaches import get_approaches_for_airport
        from services.airport_lookup import to_icao

        approach_count = 0

        def replace_approach(m):
            nonlocal approach_count
            raw_type = m.group(1).upper()
            runway_raw = m.group(2)
            airport_raw = (m.group(3) or '').upper()

            # Fix OCR typos in approach type
            appr_type = self._fix_approach_type(raw_type)
            if not appr_type:
                return m.group(0)  # not an approach, leave unchanged

            approach_count += 1

            # Clean runway: strip trailing garbage (Z, 2 etc.), keep only digits + L/C/R
            runway = re.sub(r'[^0-9LCR]$', '', runway_raw, flags=re.IGNORECASE).upper()

            # Convert airport to ICAO; fall back to entry's route_to
            icao = to_icao(airport_raw) if airport_raw else ''
            if not icao and entry:
                icao = (entry.get('route_to') or '').upper()

            # Try matching against official approaches
            # If initial icao doesn't yield a match, also try entry's route_to
            candidates = [icao] if icao else []
            if entry and (entry.get('route_to') or '').upper() not in candidates:
                candidates.append((entry.get('route_to') or '').upper())

            approach_id = f"{appr_type}{runway}"
            for try_icao in candidates:
                if not try_icao:
                    continue
                approaches = get_approaches_for_airport(try_icao)

                # Find matching official approach
                for appr in approaches:
                    if appr['id'].upper() == approach_id.upper():
                        print(f"Approach match: '{m.group(0)}' → '{appr['name']} {try_icao}'")
                        return f"{appr['name']} {try_icao}"

                # No exact match — try matching just by runway number (ignore L/C/R suffix issues)
                runway_num = re.match(r'(\d+)', runway)
                if runway_num:
                    for appr in approaches:
                        if (appr['type'].upper() == appr_type and
                                appr['id'].upper().startswith(appr_type) and
                                runway_num.group(1) in appr['id']):
                            appr_rwy = re.search(r'(\d+[LCR]?)', appr['id'][len(appr_type):])
                            if appr_rwy and appr_rwy.group(1).startswith(runway_num.group(1)):
                                print(f"Approach fuzzy match: '{m.group(0)}' → '{appr['name']} {try_icao}'")
                                return f"{appr['name']} {try_icao}"

            icao = candidates[0] if candidates else ''

            # Fallback: format cleanly without official name lookup
            result = f"{appr_type} {runway}"
            if icao:
                result += f" {icao}"
            elif airport_raw:
                result += f" {airport_raw}"
            return result

        normalized = self._APPROACH_RE.sub(replace_approach, remarks)

        # Update approach count on the entry
        if entry is not None and approach_count > 0:
            entry['num_inst_app'] = approach_count

        return normalized

    def _clean_remarks(self, entries: list[dict]) -> list[dict]:
        """Clean up garbled/unreadable remarks text.

        Retains readable aviation text, clears obviously garbled gibberish.
        """
        for entry in entries:
            remarks = entry.get('remarks', '').strip()
            if not remarks:
                continue

            # Check for non-ASCII characters (Korean, Japanese, etc.) - sign of OCR failure
            ascii_chars = sum(1 for c in remarks if ord(c) < 128)
            if len(remarks) > 0 and ascii_chars / len(remarks) < 0.8:
                print(f"Remarks cleaned (non-ASCII): '{remarks[:50]}...'")
                entry['remarks'] = ''
                continue

            # Very short garbled strings that are just numbers/symbols
            if len(remarks) <= 3 and not re.search(r'[a-zA-Z]{2,}', remarks):
                entry['remarks'] = ''
                continue

            # Normalize approach notation: match against official approach names
            # "ILS16RZ PAE" → "ILS RWY 16R KPAE", "RNAV 03 SEZ" → "RNAV (GPS) RWY 03 KSEZ"
            remarks = self._normalize_approach_in_remarks(remarks, entry)
            entry['remarks'] = remarks.strip()

        return entries

    def _fix_landings(self, entries: list[dict]) -> list[dict]:
        """Fix landing counts for single-leg flights.

        Common OCR misread: handwritten "1" read as "2".
        For a simple A→B flight (no multi-leg route, no touch-and-go indicators),
        landings should be exactly 1 (day or night, not both > 0 typically).
        """
        TOUCH_AND_GO_PATTERNS = re.compile(
            r'T\s*&?\s*G|touch\s*and\s*go|TnG|pattern|traffic\s*pattern',
            re.IGNORECASE
        )

        for entry in entries:
            route_via = entry.get('route_via', '').strip()
            remarks = entry.get('remarks', '').strip()
            landings_day = int(entry.get('landings_day', 0) or 0)
            landings_night = int(entry.get('landings_night', 0) or 0)
            total_landings = landings_day + landings_night

            # Only fix single-leg flights (no route_via = no intermediate stops)
            if route_via:
                continue

            # Skip if landings look reasonable (0 or 1)
            if total_landings <= 1:
                continue

            # Skip if remarks suggest training / touch-and-goes
            if TOUCH_AND_GO_PATTERNS.search(remarks):
                continue

            # Skip if route_from == route_to (local pattern work, multiple landings expected)
            route_from = entry.get('route_from', '').strip()
            route_to = entry.get('route_to', '').strip()
            if route_from and route_to and route_from == route_to:
                continue

            # Single-leg A→B flight with >1 landings and no T&G — likely OCR error
            if landings_day > 1 and landings_night == 0:
                print(f"Landing fix: {route_from}→{route_to} landings_day {landings_day}→1 (single-leg)")
                entry['landings_day'] = 1
            elif landings_night > 1 and landings_day == 0:
                print(f"Landing fix: {route_from}→{route_to} landings_night {landings_night}→1 (single-leg)")
                entry['landings_night'] = 1

        return entries

    def _normalize_entry(self, entry: dict) -> dict:
        """Normalize a flight entry to ensure all fields exist with proper types."""
        def to_float(val):
            if val is None:
                return 0.0
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        def to_int(val):
            if val is None:
                return 0
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0

        return {
            'date': str(entry.get('date', '') or ''),
            'aircraft_model': str(entry.get('aircraft_model', '') or ''),
            'aircraft_ident': str(entry.get('aircraft_ident', '') or ''),
            'route_from': str(entry.get('route_from', '') or ''),
            'route_to': str(entry.get('route_to', '') or ''),
            'route_via': str(entry.get('route_via', '') or ''),
            'remarks': str(entry.get('remarks', '') or ''),
            'total_duration': to_float(entry.get('total_duration')),
            'pic': to_float(entry.get('pic')),
            'sic': to_float(entry.get('sic')),
            'dual_recd': to_float(entry.get('dual_recd')),
            'dual_given': to_float(entry.get('dual_given')),
            'solo': to_float(entry.get('solo')),
            'cross_country': to_float(entry.get('cross_country')),
            'night': to_float(entry.get('night')),
            'actual_inst': to_float(entry.get('actual_inst')),
            'simulated_inst': to_float(entry.get('simulated_inst')),
            'num_inst_app': to_int(entry.get('num_inst_app')),
            'landings_day': to_int(entry.get('landings_day')),
            'landings_night': to_int(entry.get('landings_night')),
            'sel': to_float(entry.get('sel')),
            'mel': to_float(entry.get('mel')),
            'sim': to_float(entry.get('sim')),
        }

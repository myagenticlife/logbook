"""Logbook entry data model - ASA Standard Pilot Logbook format."""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional


def normalize_airport(code: str) -> str:
    """Normalize airport code to ICAO format for comparison and storage.

    SQL → KSQL, DUT → PADU, KSQL → KSQL, KBFI* → KBFI.
    Uses the airports.csv database for accurate mapping.
    """
    if not code:
        return code
    from services.airport_lookup import to_icao
    return to_icao(code)


def make_entry_key(date: str, route_from: str, route_to: str) -> str:
    """Build a normalized duplicate-detection key using ICAO codes."""
    return f"{date}|{normalize_airport(route_from)}|{normalize_airport(route_to)}"


@dataclass
class LogbookEntry:
    """ASA Standard Pilot Logbook entry format."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Basic flight info
    date: str = ""  # MM/DD/YYYY
    aircraft_model: str = ""  # Make & Model
    aircraft_ident: str = ""  # N-number
    route_from: str = ""  # Departure
    route_to: str = ""  # Arrival
    route_via: str = ""  # Full multi-leg route (e.g., "BFI-PAE-BFI"), empty for simple flights

    # Aircraft Category
    sel: float = 0.0  # Single Engine Land
    mel: float = 0.0  # Multi Engine Land

    # Conditions of Flight
    day: float = 0.0
    night: float = 0.0
    cross_country: float = 0.0
    actual_inst: float = 0.0  # Actual Instrument
    simulated_inst: float = 0.0  # Simulated Instrument (Hood)

    # Instrument
    num_inst_app: int = 0  # Number of Instrument Approaches

    # Takeoffs and landings
    landings_day: int = 0
    landings_night: int = 0
    takeoffs_day: int = 0
    takeoffs_night: int = 0

    # Type of Piloting Time
    pic: float = 0.0  # Pilot in Command
    sic: float = 0.0  # Second in Command
    dual_recd: float = 0.0  # Dual Received
    dual_given: float = 0.0  # Dual Given (CFI)
    solo: float = 0.0
    sim: float = 0.0  # Simulator / FTD time

    # Total Duration
    total_duration: float = 0.0
    duration_estimated: bool = False  # True if duration was estimated (needs verification)

    # Remarks
    remarks: str = ""

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Entry state
    locked: bool = False
    reviewed: bool = True  # Manually created entries don't need review
    needs_review: bool = False  # Agent flags entries it wants the pilot to review
    review_note: str = ""       # Agent's comment explaining what to check
    source: str = "manual"  # "manual", "scan", "flightaware", "sheets"

    # Owner
    user_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LogbookEntry":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "LogbookEntry":
        return cls.from_dict(json.loads(json_str))

    def update_timestamp(self):
        self.updated_at = datetime.utcnow().isoformat()

    def to_sheets_row(self) -> list:
        return [
            self.date,
            self.aircraft_model,
            self.aircraft_ident,
            self.route_from,
            self.route_to,
            self.route_via or "",
            self.sel or "",
            self.mel or "",
            self.day or "",
            self.night or "",
            self.cross_country or "",
            self.actual_inst or "",
            self.simulated_inst or "",
            self.num_inst_app or "",
            self.landings_day or "",
            self.landings_night or "",
            self.pic or "",
            self.sic or "",
            self.dual_recd or "",
            self.dual_given or "",
            self.solo or "",
            self.sim or "",
            self.total_duration or "",
            self.remarks,
            self.source or "manual",
        ]

    @staticmethod
    def get_sheets_headers() -> list:
        return [
            "Date",
            "Aircraft Make & Model",
            "Aircraft Ident",
            "From",
            "To",
            "Route",
            "SEL",
            "MEL",
            "Day",
            "Night",
            "Cross Country",
            "Actual Inst",
            "Sim Inst",
            "# Appr",
            "Day Ldg",
            "Night Ldg",
            "PIC",
            "SIC",
            "Dual Recd",
            "Dual Given",
            "Solo",
            "Sim/FTD",
            "Total",
            "Remarks",
            "Source",
        ]


class Logbook:
    """Collection of logbook entries."""

    def __init__(self):
        self.entries: dict[str, LogbookEntry] = {}

    def add_entry(self, entry: LogbookEntry) -> str:
        self.entries[entry.id] = entry
        return entry.id

    def get_entry(self, entry_id: str) -> Optional[LogbookEntry]:
        return self.entries.get(entry_id)

    def update_entry(self, entry: LogbookEntry) -> bool:
        if entry.id in self.entries:
            entry.update_timestamp()
            self.entries[entry.id] = entry
            return True
        return False

    def delete_entry(self, entry_id: str) -> bool:
        if entry_id in self.entries:
            del self.entries[entry_id]
            return True
        return False

    def get_all_entries(self, sort_by_date: bool = True) -> list[LogbookEntry]:
        entries = list(self.entries.values())
        if sort_by_date:
            # Sort by date, converting MM/DD/YYYY to datetime for proper chronological sorting
            def date_key(entry):
                try:
                    # Parse MM/DD/YYYY format
                    return datetime.strptime(entry.date, "%m/%d/%Y")
                except (ValueError, AttributeError):
                    # If date is invalid or missing, put it at the end
                    return datetime.min
            entries.sort(key=date_key, reverse=True)
        return entries

    def get_most_recent_flight_date(self) -> Optional[datetime]:
        """
        Get the date of the most recent flight in the logbook.

        Returns:
            datetime object of the most recent flight, or None if logbook is empty
        """
        if not self.entries:
            return None

        most_recent = None
        for entry in self.entries.values():
            try:
                entry_date = datetime.strptime(entry.date, "%m/%d/%Y")
                if most_recent is None or entry_date > most_recent:
                    most_recent = entry_date
            except (ValueError, AttributeError):
                # Skip invalid dates
                continue

        return most_recent

    def get_totals(self) -> dict:
        totals = {
            "sel": 0.0,
            "mel": 0.0,
            "day": 0.0,
            "night": 0.0,
            "cross_country": 0.0,
            "actual_inst": 0.0,
            "simulated_inst": 0.0,
            "num_inst_app": 0,
            "landings_day": 0,
            "landings_night": 0,
            "pic": 0.0,
            "sic": 0.0,
            "dual_recd": 0.0,
            "dual_given": 0.0,
            "solo": 0.0,
            "sim": 0.0,
            "total_duration": 0.0,
            "flights": len(self.entries),
        }
        for entry in self.entries.values():
            totals["sel"] += entry.sel
            totals["mel"] += entry.mel
            totals["day"] += entry.day
            totals["night"] += entry.night
            totals["cross_country"] += entry.cross_country
            totals["actual_inst"] += entry.actual_inst
            totals["simulated_inst"] += entry.simulated_inst
            totals["num_inst_app"] += entry.num_inst_app
            totals["landings_day"] += entry.landings_day
            totals["landings_night"] += entry.landings_night
            totals["pic"] += entry.pic
            totals["sic"] += entry.sic
            totals["dual_recd"] += entry.dual_recd
            totals["dual_given"] += entry.dual_given
            totals["solo"] += entry.solo
            totals["sim"] += entry.sim
            totals["total_duration"] += entry.total_duration
        return totals

    def to_json(self) -> str:
        return json.dumps(
            {id: entry.to_dict() for id, entry in self.entries.items()},
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Logbook":
        logbook = cls()
        data = json.loads(json_str)
        for entry_id, entry_data in data.items():
            entry = LogbookEntry.from_dict(entry_data)
            logbook.entries[entry_id] = entry
        return logbook

    def save_to_file(self, filepath: str):
        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load_from_file(cls, filepath: str) -> "Logbook":
        try:
            with open(filepath, "r") as f:
                return cls.from_json(f.read())
        except FileNotFoundError:
            return cls()

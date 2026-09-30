"""
PostgreSQL storage service for pilot logbook.

Drop-in replacement for SQLiteStorage, used when DATABASE_URL is configured.
"""

import os
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import psycopg2
import psycopg2.extras

from models.logbook_entry import LogbookEntry
from models.user import User


class PostgresStorage:
    """Thread-safe PostgreSQL storage for logbook entries."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._local = threading.local()
        self._db_initialized = False
        print("PostgresStorage created (lazy init — DB schema created on first request)")

    @contextmanager
    def _get_connection(self):
        """Get thread-local database connection."""
        if not self._db_initialized:
            self._db_initialized = True  # prevent recursion
            try:
                self._init_db()
            except Exception as e:
                self._db_initialized = False
                print(f"DB init retry failed: {e}")
                raise
        if not hasattr(self._local, "conn") or self._local.conn is None or self._local.conn.closed:
            self._local.conn = psycopg2.connect(
                self.database_url,
                cursor_factory=psycopg2.extras.RealDictCursor,
                connect_timeout=10,
            )
            self._local.conn.autocommit = False
        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS entries (
                        id TEXT PRIMARY KEY,
                        date TEXT,
                        aircraft_model TEXT,
                        aircraft_ident TEXT,
                        route_from TEXT,
                        route_to TEXT,
                        route_via TEXT DEFAULT '',
                        sel DOUBLE PRECISION DEFAULT 0,
                        mel DOUBLE PRECISION DEFAULT 0,
                        day DOUBLE PRECISION DEFAULT 0,
                        night DOUBLE PRECISION DEFAULT 0,
                        cross_country DOUBLE PRECISION DEFAULT 0,
                        actual_inst DOUBLE PRECISION DEFAULT 0,
                        simulated_inst DOUBLE PRECISION DEFAULT 0,
                        num_inst_app INTEGER DEFAULT 0,
                        landings_day INTEGER DEFAULT 0,
                        landings_night INTEGER DEFAULT 0,
                        takeoffs_day INTEGER DEFAULT 0,
                        takeoffs_night INTEGER DEFAULT 0,
                        pic DOUBLE PRECISION DEFAULT 0,
                        sic DOUBLE PRECISION DEFAULT 0,
                        dual_recd DOUBLE PRECISION DEFAULT 0,
                        dual_given DOUBLE PRECISION DEFAULT 0,
                        solo DOUBLE PRECISION DEFAULT 0,
                        sim DOUBLE PRECISION DEFAULT 0,
                        total_duration DOUBLE PRECISION DEFAULT 0,
                        duration_estimated INTEGER DEFAULT 0,
                        remarks TEXT DEFAULT '',
                        created_at TEXT,
                        updated_at TEXT,
                        locked INTEGER DEFAULT 0,
                        reviewed INTEGER DEFAULT 1,
                        needs_review INTEGER DEFAULT 0,
                        review_note TEXT DEFAULT '',
                        user_id TEXT DEFAULT ''
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT DEFAULT '',
                        name TEXT NOT NULL DEFAULT '',
                        google_id TEXT UNIQUE,
                        avatar_url TEXT DEFAULT '',
                        default_tail_number TEXT DEFAULT '',
                        default_aircraft_type TEXT DEFAULT '',
                        default_departure TEXT DEFAULT '',
                        created_at TEXT,
                        updated_at TEXT
                    )
                """)

                # Create indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_date ON entries(date)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_aircraft ON entries(aircraft_ident)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON entries(user_id)")

            conn.commit()

            # Migrate: add new entry columns if they don't exist
            for col, col_def in [
                ("source", "TEXT DEFAULT 'manual'"),
                ("takeoffs_day", "INTEGER DEFAULT 0"),
                ("takeoffs_night", "INTEGER DEFAULT 0"),
                ("needs_review", "INTEGER DEFAULT 0"),
                ("review_note", "TEXT DEFAULT ''"),
            ]:
                try:
                    with conn.cursor() as cur:
                        cur.execute(f"ALTER TABLE entries ADD COLUMN {col} {col_def}")
                    conn.commit()
                except psycopg2.errors.DuplicateColumn:
                    conn.rollback()

            # Migrate: add new user columns if they don't exist
            for col, col_def in [
                ("google_refresh_token", "TEXT DEFAULT ''"),
                ("backup_sheet_id", "TEXT DEFAULT ''"),
                ("access_token", "TEXT DEFAULT ''"),
            ]:
                try:
                    with conn.cursor() as cur:
                        cur.execute(f"ALTER TABLE users ADD COLUMN {col} {col_def}")
                    conn.commit()
                except psycopg2.errors.DuplicateColumn:
                    conn.rollback()

            try:
                with conn.cursor() as cur:
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_access_token ON users(access_token)")
                conn.commit()
            except Exception:
                conn.rollback()

        # Normalize existing airport codes to ICAO
        try:
            self._normalize_airport_codes()
        except Exception as e:
            print(f"ICAO normalization skipped: {e}")

    def _normalize_airport_codes(self):
        """One-time migration: convert all airport codes to ICAO format."""
        from services.airport_lookup import to_icao

        with self._get_connection() as conn:
            # Quick check: any 3-letter codes left?
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) as cnt FROM entries WHERE "
                    "(LENGTH(route_from) = 3 AND route_from ~ '^[A-Z]+$') OR "
                    "(LENGTH(route_to) = 3 AND route_to ~ '^[A-Z]+$')"
                )
                if cur.fetchone()["cnt"] == 0:
                    return

            with conn.cursor() as cur:
                cur.execute("SELECT id, route_from, route_to, route_via FROM entries")
                rows = cur.fetchall()

            updated = 0
            for row in rows:
                rf = row["route_from"] or ""
                rt = row["route_to"] or ""
                rv = row["route_via"] or ""
                new_rf = to_icao(rf) if rf else rf
                new_rt = to_icao(rt) if rt else rt
                # Normalize route_via legs
                if rv:
                    legs = [to_icao(leg.strip()) for leg in rv.split("-") if leg.strip()]
                    new_rv = "-".join(legs)
                else:
                    new_rv = rv

                if new_rf != rf or new_rt != rt or new_rv != rv:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE entries SET route_from=%s, route_to=%s, route_via=%s WHERE id=%s",
                            (new_rf, new_rt, new_rv, row["id"]),
                        )
                    updated += 1

            if updated:
                conn.commit()
                print(f"ICAO normalization: updated {updated} entries")

    def add_entry(self, entry: LogbookEntry, user_id: str = "") -> str:
        """Add a new entry to the database."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO entries (
                        id, date, aircraft_model, aircraft_ident,
                        route_from, route_to, route_via, sel, mel, day, night,
                        cross_country, actual_inst, simulated_inst,
                        num_inst_app, landings_day, landings_night,
                        takeoffs_day, takeoffs_night,
                        pic, sic, dual_recd, dual_given, solo, sim,
                        total_duration, duration_estimated, remarks,
                        created_at, updated_at, locked, reviewed,
                        needs_review, review_note, source, user_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        entry.id,
                        entry.date,
                        entry.aircraft_model,
                        entry.aircraft_ident,
                        entry.route_from,
                        entry.route_to,
                        entry.route_via,
                        entry.sel,
                        entry.mel,
                        entry.day,
                        entry.night,
                        entry.cross_country,
                        entry.actual_inst,
                        entry.simulated_inst,
                        entry.num_inst_app,
                        entry.landings_day,
                        entry.landings_night,
                        entry.takeoffs_day,
                        entry.takeoffs_night,
                        entry.pic,
                        entry.sic,
                        entry.dual_recd,
                        entry.dual_given,
                        entry.solo,
                        entry.sim,
                        entry.total_duration,
                        1 if entry.duration_estimated else 0,
                        entry.remarks,
                        entry.created_at,
                        entry.updated_at,
                        1 if entry.locked else 0,
                        1 if entry.reviewed else 0,
                        1 if entry.needs_review else 0,
                        entry.review_note,
                        entry.source,
                        user_id,
                    ),
                )
            conn.commit()
        return entry.id

    def get_entry(self, entry_id: str, user_id: str = "") -> Optional[LogbookEntry]:
        """Get a single entry by ID, scoped to user."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM entries WHERE id = %s AND user_id = %s",
                    (entry_id, user_id),
                )
                row = cur.fetchone()
                if row:
                    return self._row_to_entry(row)
        return None

    def get_all_entries(self, user_id: str = "", sort_by_date: bool = True) -> list[LogbookEntry]:
        """Get all entries for a user, optionally sorted by date (most recent first)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if sort_by_date:
                    # Parse MM/DD/YYYY dates for sorting
                    cur.execute("""
                        SELECT * FROM entries WHERE user_id = %s
                        ORDER BY
                            CASE
                                WHEN date ~ '^[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}$'
                                THEN TO_DATE(date, 'MM/DD/YYYY')
                                ELSE TO_DATE('01/01/1900', 'MM/DD/YYYY')
                            END DESC
                    """, (user_id,))
                else:
                    cur.execute("SELECT * FROM entries WHERE user_id = %s", (user_id,))
                return [self._row_to_entry(row) for row in cur.fetchall()]

    def update_entry(self, entry: LogbookEntry, user_id: str = "") -> bool:
        """Update an existing entry, scoped to user."""
        entry.update_timestamp()
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE entries SET
                        date=%s, aircraft_model=%s, aircraft_ident=%s,
                        route_from=%s, route_to=%s, route_via=%s,
                        sel=%s, mel=%s, day=%s, night=%s,
                        cross_country=%s, actual_inst=%s, simulated_inst=%s,
                        num_inst_app=%s, landings_day=%s, landings_night=%s,
                        takeoffs_day=%s, takeoffs_night=%s,
                        pic=%s, sic=%s, dual_recd=%s, dual_given=%s, solo=%s, sim=%s,
                        total_duration=%s, duration_estimated=%s, remarks=%s,
                        updated_at=%s, locked=%s, reviewed=%s,
                        needs_review=%s, review_note=%s, source=%s
                    WHERE id=%s AND user_id=%s
                """,
                    (
                        entry.date,
                        entry.aircraft_model,
                        entry.aircraft_ident,
                        entry.route_from,
                        entry.route_to,
                        entry.route_via,
                        entry.sel,
                        entry.mel,
                        entry.day,
                        entry.night,
                        entry.cross_country,
                        entry.actual_inst,
                        entry.simulated_inst,
                        entry.num_inst_app,
                        entry.landings_day,
                        entry.landings_night,
                        entry.takeoffs_day,
                        entry.takeoffs_night,
                        entry.pic,
                        entry.sic,
                        entry.dual_recd,
                        entry.dual_given,
                        entry.solo,
                        entry.sim,
                        entry.total_duration,
                        1 if entry.duration_estimated else 0,
                        entry.remarks,
                        entry.updated_at,
                        1 if entry.locked else 0,
                        1 if entry.reviewed else 0,
                        1 if entry.needs_review else 0,
                        entry.review_note,
                        entry.source,
                        entry.id,
                        user_id,
                    ),
                )
            conn.commit()
            return cur.rowcount > 0

    def delete_entry(self, entry_id: str, user_id: str = "") -> bool:
        """Delete an entry by ID, scoped to user. Refuses to delete locked entries."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM entries WHERE id = %s AND user_id = %s AND locked = 0",
                    (entry_id, user_id),
                )
            conn.commit()
            return cur.rowcount > 0

    def delete_entries(self, entry_ids: list[str], user_id: str = "") -> dict:
        """Delete multiple entries, scoped to user, skipping locked ones."""
        if not entry_ids:
            return {"deleted": 0, "skipped_locked": 0}
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                placeholders = ",".join(["%s"] * len(entry_ids))
                params = entry_ids + [user_id]
                cur.execute(
                    f"SELECT COUNT(*) as cnt FROM entries WHERE id IN ({placeholders}) AND user_id = %s AND locked = 1",
                    params,
                )
                skipped = cur.fetchone()["cnt"]
                cur.execute(
                    f"DELETE FROM entries WHERE id IN ({placeholders}) AND user_id = %s AND locked = 0",
                    params,
                )
            conn.commit()
            return {"deleted": cur.rowcount, "skipped_locked": skipped}

    def toggle_entry_field(self, entry_id: str, field: str, value: bool, user_id: str = "") -> bool:
        """Toggle a boolean field (locked/reviewed) on an entry, scoped to user."""
        if field not in ("locked", "reviewed"):
            raise ValueError(f"Cannot toggle field: {field}")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Use format for column name (safe - validated above), %s for values
                cur.execute(
                    f"UPDATE entries SET {field} = %s, updated_at = %s WHERE id = %s AND user_id = %s",
                    (1 if value else 0, datetime.utcnow().isoformat(), entry_id, user_id),
                )
            conn.commit()
            return cur.rowcount > 0

    def get_totals(self, user_id: str = "") -> dict:
        """Calculate totals for a user using SQL aggregation."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        COALESCE(SUM(sel), 0) as sel,
                        COALESCE(SUM(mel), 0) as mel,
                        COALESCE(SUM(day), 0) as day,
                        COALESCE(SUM(night), 0) as night,
                        COALESCE(SUM(cross_country), 0) as cross_country,
                        COALESCE(SUM(actual_inst), 0) as actual_inst,
                        COALESCE(SUM(simulated_inst), 0) as simulated_inst,
                        COALESCE(SUM(num_inst_app), 0) as num_inst_app,
                        COALESCE(SUM(landings_day), 0) as landings_day,
                        COALESCE(SUM(landings_night), 0) as landings_night,
                        COALESCE(SUM(pic), 0) as pic,
                        COALESCE(SUM(sic), 0) as sic,
                        COALESCE(SUM(dual_recd), 0) as dual_recd,
                        COALESCE(SUM(dual_given), 0) as dual_given,
                        COALESCE(SUM(solo), 0) as solo,
                        COALESCE(SUM(sim), 0) as sim,
                        COALESCE(SUM(total_duration), 0) as total_duration,
                        COUNT(*) as flights
                    FROM entries WHERE user_id = %s
                """, (user_id,))
                row = cur.fetchone()
                # Convert Decimal types to float for JSON serialization
                return {k: float(v) if v is not None else 0 for k, v in row.items()}

    def get_most_recent_flight_date(self, user_id: str = "", source: str = "") -> Optional[datetime]:
        """Get the most recent flight date for a user, optionally filtered by source."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if source:
                    cur.execute("""
                        SELECT date FROM entries
                        WHERE user_id = %s AND date != '' AND date IS NOT NULL
                            AND date ~ '^[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}$'
                            AND source = %s
                        ORDER BY TO_DATE(date, 'MM/DD/YYYY') DESC
                        LIMIT 1
                    """, (user_id, source))
                else:
                    cur.execute("""
                        SELECT date FROM entries
                        WHERE user_id = %s AND date != '' AND date IS NOT NULL
                            AND date ~ '^[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}$'
                        ORDER BY TO_DATE(date, 'MM/DD/YYYY') DESC
                        LIMIT 1
                    """, (user_id,))
                row = cur.fetchone()
                if row and row["date"]:
                    try:
                        return datetime.strptime(row["date"], "%m/%d/%Y")
                    except ValueError:
                        pass
        return None

    def get_known_values(self, user_id: str = "") -> tuple[set[str], set[str], set[str]]:
        """Get known aircraft idents, models, and airports for a user."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT aircraft_ident FROM entries WHERE user_id = %s AND aircraft_ident != '' AND aircraft_ident IS NOT NULL",
                    (user_id,),
                )
                idents = {row["aircraft_ident"] for row in cur.fetchall()}

                cur.execute(
                    "SELECT DISTINCT aircraft_model FROM entries WHERE user_id = %s AND aircraft_model != '' AND aircraft_model IS NOT NULL",
                    (user_id,),
                )
                models = {row["aircraft_model"] for row in cur.fetchall()}

                cur.execute(
                    "SELECT DISTINCT route_from AS val FROM entries WHERE user_id = %s AND route_from != '' AND route_from IS NOT NULL "
                    "UNION SELECT DISTINCT route_to AS val FROM entries WHERE user_id = %s AND route_to != '' AND route_to IS NOT NULL",
                    (user_id, user_id),
                )
                airports = {row["val"] for row in cur.fetchall()}

                return idents, models, airports

    def get_existing_keys(self, user_id: str = "") -> dict[str, dict]:
        """Get dict of normalized date|from|to keys → {id, source} for duplicate detection."""
        from models.logbook_entry import make_entry_key
        keys = {}
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, date, route_from, route_to, source FROM entries WHERE user_id = %s",
                    (user_id,),
                )
                for row in cur.fetchall():
                    key = make_entry_key(row["date"], row["route_from"], row["route_to"])
                    keys[key] = {"id": row["id"], "source": row.get("source") or "manual"}
        return keys

    def claim_orphaned_entries(self, user_id: str) -> int:
        """Assign all entries with empty user_id to a user. Returns count claimed."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE entries SET user_id = %s WHERE user_id = '' OR user_id IS NULL",
                    (user_id,),
                )
            conn.commit()
            return cur.rowcount

    def bulk_insert(self, entries: list[LogbookEntry]) -> int:
        """Bulk insert entries (for migration). Returns count inserted."""
        count = 0
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for entry in entries:
                    try:
                        cur.execute(
                            """
                            INSERT INTO entries (
                                id, date, aircraft_model, aircraft_ident,
                                route_from, route_to, route_via, sel, mel, day, night,
                                cross_country, actual_inst, simulated_inst,
                                num_inst_app, landings_day, landings_night,
                                takeoffs_day, takeoffs_night,
                                pic, sic, dual_recd, dual_given, solo, sim,
                                total_duration, duration_estimated, remarks,
                                created_at, updated_at, locked, reviewed
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET
                                date=EXCLUDED.date, aircraft_model=EXCLUDED.aircraft_model,
                                aircraft_ident=EXCLUDED.aircraft_ident, route_from=EXCLUDED.route_from,
                                route_to=EXCLUDED.route_to, updated_at=EXCLUDED.updated_at
                        """,
                            (
                                entry.id,
                                entry.date,
                                entry.aircraft_model,
                                entry.aircraft_ident,
                                entry.route_from,
                                entry.route_to,
                                entry.route_via,
                                entry.sel,
                                entry.mel,
                                entry.day,
                                entry.night,
                                entry.cross_country,
                                entry.actual_inst,
                                entry.simulated_inst,
                                entry.num_inst_app,
                                entry.landings_day,
                                entry.landings_night,
                                entry.takeoffs_day,
                                entry.takeoffs_night,
                                entry.pic,
                                entry.sic,
                                entry.dual_recd,
                                entry.dual_given,
                                entry.solo,
                                entry.sim,
                                entry.total_duration,
                                1 if entry.duration_estimated else 0,
                                entry.remarks,
                                entry.created_at,
                                entry.updated_at,
                                1 if entry.locked else 0,
                                1 if entry.reviewed else 0,
                            ),
                        )
                        count += 1
                    except psycopg2.IntegrityError:
                        conn.rollback()
            conn.commit()
        return count

    def clear_all(self):
        """Clear all entries."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM entries")
            conn.commit()

    # ============== User Methods ==============

    def create_user(self, user: User) -> str:
        """Create a new user. Returns user ID."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO users (
                        id, email, password_hash, name, google_id, avatar_url,
                        google_refresh_token, backup_sheet_id,
                        default_tail_number, default_aircraft_type, default_departure,
                        access_token, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        user.id, user.email, user.password_hash, user.name,
                        user.google_id or None, user.avatar_url,
                        user.google_refresh_token, user.backup_sheet_id,
                        user.default_tail_number, user.default_aircraft_type,
                        user.default_departure, user.access_token,
                        user.created_at, user.updated_at,
                    ),
                )
            conn.commit()
        return user.id

    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                row = cur.fetchone()
                return self._row_to_user(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email (case-insensitive)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM users WHERE LOWER(email) = LOWER(%s)", (email,)
                )
                row = cur.fetchone()
                return self._row_to_user(row) if row else None

    def get_user_by_google_id(self, google_id: str) -> Optional[User]:
        """Get a user by Google OAuth ID."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM users WHERE google_id = %s", (google_id,)
                )
                row = cur.fetchone()
                return self._row_to_user(row) if row else None

    def get_user_by_access_token(self, token: str) -> Optional[User]:
        """Get a user by their access token."""
        if not token:
            return None
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM users WHERE access_token = %s", (token,)
                )
                row = cur.fetchone()
                return self._row_to_user(row) if row else None

    def list_users(self) -> list[User]:
        """List all users (admin use)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users ORDER BY created_at")
                rows = cur.fetchall()
                return [self._row_to_user(row) for row in rows]

    def update_user(self, user: User) -> bool:
        """Update a user's profile."""
        user.updated_at = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE users SET
                        email=%s, password_hash=%s, name=%s, google_id=%s, avatar_url=%s,
                        google_refresh_token=%s, backup_sheet_id=%s,
                        default_tail_number=%s, default_aircraft_type=%s, default_departure=%s,
                        access_token=%s, updated_at=%s
                    WHERE id=%s""",
                    (
                        user.email, user.password_hash, user.name,
                        user.google_id or None, user.avatar_url,
                        user.google_refresh_token, user.backup_sheet_id,
                        user.default_tail_number, user.default_aircraft_type,
                        user.default_departure, user.access_token,
                        user.updated_at, user.id,
                    ),
                )
            conn.commit()
            return cur.rowcount > 0

    def delete_user(self, user_id: str) -> bool:
        """Delete a user and all their entries."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM entries WHERE user_id = %s", (user_id,))
                cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
                conn.commit()
                return cur.rowcount > 0

    def _row_to_user(self, row: dict) -> User:
        """Convert a database row to User."""
        return User(
            id=row["id"],
            email=row["email"],
            password_hash=row["password_hash"] or "",
            name=row["name"] or "",
            google_id=row["google_id"] or "",
            avatar_url=row["avatar_url"] or "",
            google_refresh_token=row.get("google_refresh_token") or "",
            backup_sheet_id=row.get("backup_sheet_id") or "",
            default_tail_number=row["default_tail_number"] or "",
            default_aircraft_type=row["default_aircraft_type"] or "",
            default_departure=row["default_departure"] or "",
            access_token=row.get("access_token") or "",
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
        )

    def count(self) -> int:
        """Get total entry count."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as cnt FROM entries")
                return cur.fetchone()["cnt"]

    def _row_to_entry(self, row: dict) -> LogbookEntry:
        """Convert a database row to LogbookEntry."""
        return LogbookEntry(
            id=row["id"],
            date=row["date"] or "",
            aircraft_model=row["aircraft_model"] or "",
            aircraft_ident=row["aircraft_ident"] or "",
            route_from=row["route_from"] or "",
            route_to=row["route_to"] or "",
            route_via=row.get("route_via") or "",
            sel=float(row["sel"] or 0),
            mel=float(row["mel"] or 0),
            day=float(row["day"] or 0),
            night=float(row["night"] or 0),
            cross_country=float(row["cross_country"] or 0),
            actual_inst=float(row["actual_inst"] or 0),
            simulated_inst=float(row["simulated_inst"] or 0),
            num_inst_app=int(row["num_inst_app"] or 0),
            landings_day=int(row["landings_day"] or 0),
            landings_night=int(row["landings_night"] or 0),
            takeoffs_day=int((row["takeoffs_day"] if "takeoffs_day" in row else 0) or 0),
            takeoffs_night=int((row["takeoffs_night"] if "takeoffs_night" in row else 0) or 0),
            pic=float(row["pic"] or 0),
            sic=float(row["sic"] or 0),
            dual_recd=float(row["dual_recd"] or 0),
            dual_given=float(row["dual_given"] or 0),
            solo=float(row["solo"] or 0),
            sim=float(row["sim"] or 0),
            total_duration=float(row["total_duration"] or 0),
            duration_estimated=bool(row["duration_estimated"]),
            remarks=row["remarks"] or "",
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
            locked=bool(row["locked"]) if row["locked"] is not None else False,
            reviewed=bool(row["reviewed"]) if row["reviewed"] is not None else True,
            needs_review=bool(row["needs_review"]) if ("needs_review" in row and row["needs_review"] is not None) else False,
            review_note=(row["review_note"] if "review_note" in row else "") or "",
            source=row.get("source") or "manual",
        )

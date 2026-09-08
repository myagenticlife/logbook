"""Token-based access blueprint.

Each user has a personal access token (xxxx-xxxx-xxxx-xxxx). Users paste it
at /access to sign in. The admin (matching ADMIN_EMAIL env var) can manage
users and tokens at /admin.
"""

import os

from flask import (
    Blueprint, abort, flash, redirect, render_template, request, url_for
)
from flask_login import login_user, logout_user, current_user
from functools import wraps

from models.user import User, generate_access_token

auth_bp = Blueprint("auth", __name__)


def _get_storage():
    from app import storage
    return storage


def _admin_email() -> str:
    return (os.environ.get("ADMIN_EMAIL") or "").strip().lower()


def is_admin(user) -> bool:
    """Check if the given user is the configured admin."""
    if not user or not getattr(user, "email", ""):
        return False
    return user.email.strip().lower() == _admin_email()


def admin_required(view):
    """Decorator: require the request to be authenticated as the admin user."""
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for("auth.access", next=request.path))
        if not is_admin(current_user):
            abort(403)
        return view(*args, **kwargs)
    return wrapper


def _claim_orphans(user):
    """On first login, claim any entries with no user_id."""
    storage = _get_storage()
    claimed = storage.claim_orphaned_entries(user.id)
    if claimed:
        print(f"Claimed {claimed} orphaned entries for user {user.email}")


# ── Access token sign-in ───────────────────────────────────────

@auth_bp.route("/access", methods=["GET", "POST"])
def access():
    """Enter access token to sign in."""
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        token = (request.form.get("token") or "").strip().lower()
        user = _get_storage().get_user_by_access_token(token)
        if user:
            login_user(user, remember=True)
            _claim_orphans(user)
            next_page = request.args.get("next") or url_for("index")
            return redirect(next_page)
        flash("Invalid access token.", "error")

    return render_template("access.html")


@auth_bp.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("index"))


# ── Admin: create users and manage tokens ──────────────────────

@auth_bp.route("/admin", methods=["GET"])
@admin_required
def admin():
    users = _get_storage().list_users()
    return render_template("admin.html", users=users, admin_email=_admin_email())


@auth_bp.route("/admin/users/new", methods=["POST"])
@admin_required
def admin_create_user():
    email = (request.form.get("email") or "").strip().lower()
    name = (request.form.get("name") or "").strip()
    if not email:
        flash("Email is required.", "error")
        return redirect(url_for("auth.admin"))

    storage = _get_storage()
    if storage.get_user_by_email(email):
        flash(f"User {email} already exists.", "error")
        return redirect(url_for("auth.admin"))

    user = User(
        email=email,
        name=name or email.split("@")[0],
        access_token=generate_access_token(),
    )
    storage.create_user(user)
    flash(f"Created {email} with token {user.access_token}", "success")
    return redirect(url_for("auth.admin"))


@auth_bp.route("/admin/users/<user_id>/regenerate", methods=["POST"])
@admin_required
def admin_regenerate_token(user_id):
    storage = _get_storage()
    user = storage.get_user(user_id)
    if not user:
        flash("User not found.", "error")
        return redirect(url_for("auth.admin"))
    user.access_token = generate_access_token()
    storage.update_user(user)
    flash(f"New token for {user.email}: {user.access_token}", "success")
    return redirect(url_for("auth.admin"))


@auth_bp.route("/admin/users/<user_id>/delete", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    storage = _get_storage()
    user = storage.get_user(user_id)
    if not user:
        flash("User not found.", "error")
        return redirect(url_for("auth.admin"))
    if is_admin(user):
        flash("Cannot delete the admin user.", "error")
        return redirect(url_for("auth.admin"))
    if hasattr(storage, "delete_user"):
        storage.delete_user(user_id)
        flash(f"Deleted {user.email}.", "success")
    else:
        flash("Delete not supported by storage backend.", "error")
    return redirect(url_for("auth.admin"))


# ── Bootstrap: ensure admin user has a token at startup ────────

def ensure_admin_bootstrap(storage):
    """Create the admin user (or generate their token) on app startup.

    If ADMIN_EMAIL is set and that user doesn't have an access token,
    generate one and print it to logs so the operator can sign in.
    """
    email = _admin_email()
    if not email:
        print("AUTH BOOTSTRAP: ADMIN_EMAIL not set; skipping admin bootstrap")
        return

    user = storage.get_user_by_email(email)
    if user is None:
        user = User(
            email=email,
            name=email.split("@")[0],
            access_token=generate_access_token(),
        )
        storage.create_user(user)
        print("=" * 60)
        print(f"AUTH BOOTSTRAP: Created admin user {email}")
        print(f"  ACCESS TOKEN: {user.access_token}")
        print("=" * 60)
        return

    if not user.access_token:
        user.access_token = generate_access_token()
        storage.update_user(user)
        print("=" * 60)
        print(f"AUTH BOOTSTRAP: Generated token for admin {email}")
        print(f"  ACCESS TOKEN: {user.access_token}")
        print("=" * 60)
        return

    # Operator escape hatch: reprint the EXISTING admin token to the logs when
    # PRINT_ADMIN_TOKEN is set (for recovering access). Unset it afterward.
    if os.environ.get("PRINT_ADMIN_TOKEN"):
        print("=" * 60)
        print(f"AUTH BOOTSTRAP: Admin {email} already exists")
        print(f"  ACCESS TOKEN: {user.access_token}")
        print("=" * 60)

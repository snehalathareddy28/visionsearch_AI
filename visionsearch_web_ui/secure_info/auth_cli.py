"""Local-only user provisioning for the Silver Jubilee authentication store."""

from __future__ import annotations

import argparse
import getpass
import sys
from collections.abc import Sequence

from visionsearch_web_ui.secure_info.auth import (
    AdminAlreadyProvisionedError,
    AuthenticationStore,
    AuthenticationStoreError,
    DuplicateUserError,
)


def _read_password() -> str:
    password = getpass.getpass("Password (minimum 12 characters): ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise ValueError("The password entries did not match.")
    return password


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Provision Silver Jubilee users locally; passwords are entered without echo."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command_name, description in (
        ("create-admin", "Create the one-time initial administrator."),
        ("create-user", "Create a normal user without public self-registration."),
    ):
        command = commands.add_parser(command_name, help=description)
        command.add_argument("--username", required=True)
    arguments = parser.parse_args(argv)

    store = AuthenticationStore()
    try:
        store.initialize()
        password = _read_password()
        if arguments.command == "create-admin":
            user = store.create_admin(arguments.username, password)
        else:
            user = store.create_user(arguments.username, password)
    except (ValueError, AdminAlreadyProvisionedError, DuplicateUserError) as error:
        print(str(error), file=sys.stderr)
        return 2
    except AuthenticationStoreError:
        print("The authentication store is unavailable.", file=sys.stderr)
        return 1

    print(f"Provisioned {user.role} account '{user.username}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

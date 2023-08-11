#!/usr/bin/env bash

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd)

function print_error
{
    # shellcheck disable=SC2145
    echo -e "\033[31m$@\033[0m" 1>&2
}

function print_message
{
    # shellcheck disable=SC2145
    echo -e "\033[32m$@\033[0m"
}

trap 'cancel_black' INT

function cancel_black
{
    print_error "An interrupt signal was detected."
    exit 1
}

if ! "$ROOT_DIR/python" -m pip show -q opencv-python &> /dev/null; then
    print_error "Not found cv2 package"
    exit 1
fi

DOWNLOAD_URL="https://raw.githubusercontent.com/microsoft/python-type-stubs/main/cv2/__init__.pyi"
DEST_DIR=$("$ROOT_DIR/python" -c 'import cv2, os; print(os.path.dirname(cv2.__file__))')

curl -sSL "$DOWNLOAD_URL" -o "$DEST_DIR/__init__.pyi"

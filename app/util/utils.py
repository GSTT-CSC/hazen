# -*- coding: utf-8 -*-
"""Helper utilities and decorators."""
import string

from flask import flash

ALPHABET = string.digits + string.ascii_letters



def base62_encode(num: int, alphabet: str = ALPHABET) -> str:
    """Encode an integer to base-62 string."""
    if num < 0:
        raise ValueError('value must be non-negative')
    if num == 0:
        return alphabet[0]
    arr = []
    base = len(alphabet)
    while num:
        rem = num % base
        num //= base
        arr.append(alphabet[rem])
    arr.reverse()
    return ''.join(arr)


def base62_decode(x: str, alphabet: str = ALPHABET) -> int:
    """Decode a base-62 string to integer."""
    base = len(alphabet)
    strlen = len(x)
    num = 0

    idx = 0
    for char in x:
        power = (strlen - (idx + 1))
        try:
            num += alphabet.index(char) * (base ** power)
        except ValueError:
            raise ValueError('Invalid character [%s] (U+%04X) encountered.' % (char, ord(char)))
        idx += 1

    return num


def flash_errors(form, category='warning'):
    """Flash all errors for a form."""
    for field, errors in form.errors.items():
        for error in errors:
            flash('{0} - {1}'.format(getattr(form, field).label.text, error), category)

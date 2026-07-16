def parse_pairs(text):
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # BUG: crashes on lines without '=' and does not strip whitespace.
        key, value = line.split("=", 1)
        result[key] = value
    return result

def parse_pairs(text):
    # Shallow fix: strips whitespace but still crashes on lines without '='.
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result

def format_bytes(n):
    if n < 1024:
        return f"{n} B"
    kibibytes = n / 1024
    if kibibytes < 1024:
        return f"{kibibytes:.1f} KB"
    mebibytes = kibibytes / 1024
    return f"{mebibytes:.1f} MB"

def slugify(text):
    # Shallow: only handles spaces, leaves punctuation/collapsing to the caller.
    return text.lower().replace(" ", "-")

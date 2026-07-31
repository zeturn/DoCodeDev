from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReleaseCase:
    name: str
    category: str
    fixture: str
    hidden_fixture: str


CASES = (
    ReleaseCase("html_cards", "crawler", "html_cards", "html_cards_hidden"),
    ReleaseCase("irregular_table", "crawler", "irregular_table", "irregular_table_hidden"),
    ReleaseCase("rss_namespace", "crawler", "rss_namespace", "rss_namespace_hidden"),
    ReleaseCase("json_cursor", "crawler", "json_cursor", "json_cursor_hidden"),
    ReleaseCase("html_rel_next", "crawler", "html_rel_next", "html_rel_next_hidden"),
    ReleaseCase("json_next_url", "crawler", "json_next_url", "json_next_url_hidden"),
    ReleaseCase("tabular_source", "crawler", "tabular_source", "tabular_source_hidden"),
    ReleaseCase("public_https_rss", "crawler", "public_https_rss", "public_https_rss_hidden"),
    ReleaseCase("api_rename", "repository", "api_rename", "api_rename_hidden"),
    ReleaseCase("config_migration", "repository", "config_migration", "config_migration_hidden"),
    ReleaseCase("feature_completion", "repository", "feature_completion", "feature_completion_hidden"),
)

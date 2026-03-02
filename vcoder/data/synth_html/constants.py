"""Constants for synthetic HTML dataset generation."""

from __future__ import annotations

from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("outputs/synth_html")
DEFAULT_NUM_SAMPLES = 200
DEFAULT_SEED = 1337
PROGRESS_EVERY = 20
SIMPLE_LAYOUT_SHARE = 0.20
INDEX_FIELDS = ("sample_id", "template", "source_html", "render_png")

SERVICE_NAMES = [
    "auth-gateway",
    "billing-core",
    "risk-scoring",
    "shipment-sync",
    "analytics-api",
    "catalog-index",
    "fulfillment-bot",
]
REGIONS = ["NA", "EU", "APAC", "LATAM", "MEA"]
OWNERS = ["A. Reed", "M. Patel", "K. Sato", "L. Chen", "R. Lopez", "S. Kim"]
RISK_CODES = ["R-11", "R-23", "R-34", "R-55", "R-72", "R-88"]
TASKS = ["Schema migration", "Load test", "Rollback drill", "Access review", "Patch window"]
CHECKS = ["SAST", "DAST", "Perf", "Backup", "Failover", "Audit"]
MATERIALS = ["Lithium", "Copper", "Neon", "Silicon", "Nickel", "Graphite"]
BRAND_LOGO_LABELS = [
    "Logo Slot A",
    "Logo Slot B",
    "Brand Mark",
    "Company Seal",
    "Subsidiary Mark",
    "Product Badge",
]
BRAND_PICTURE_LABELS = [
    "Company Logo Area",
    "Partner Mark Area",
    "Sponsor Logo Box",
    "Vendor Identity Box",
    "Client Crest Zone",
    "Brand Placeholder",
]
BULKY_TEXT_SNIPPETS = [
    "Escalation pressure remains elevated across two regions.",
    "Portfolio confidence improves after capacity rebalance.",
    "Execution quality is strong but latency spikes persist.",
    "Commercial demand is stable with selective risk pockets.",
    "Coordination overhead increased during the latest cycle.",
]

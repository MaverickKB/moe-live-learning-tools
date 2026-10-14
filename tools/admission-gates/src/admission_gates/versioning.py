"""Version tags for the admission API, pocket entries and bundles.

Bump API_VERSION when a route, request field or Python entry point changes meaning.
Bump a schema version when a stored record's fields change meaning. Readers refuse
mismatched majors rather than guessing.
"""

API_VERSION = "1.0"
API_VERSION_HEADER = "X-Admission-API-Version"
SCHEMAS = {
    "pocket_entry": 1,
    "pocket_bundle": 1,
    "consolidation_bundle": 1,
    "admission_settings": 1,
}


class VersionMismatch(ValueError):
    pass


def api_major(version=API_VERSION):
    if not isinstance(version, str) or not version.split(".")[0].isdigit():
        raise VersionMismatch("api_version must look like '<major>.<minor>'")
    return int(version.split(".")[0])


def stamp(record, kind):
    """Return a copy of record carrying schema_version and api_version."""
    out = dict(record)
    out["schema_version"] = SCHEMAS[kind]
    out["api_version"] = API_VERSION
    return out


def check(record, kind):
    """Refuse a record whose schema or API major differs from this build."""
    if not isinstance(record, dict):
        raise VersionMismatch(kind + " must be an object")
    found = record.get("schema_version")
    if type(found) is not int or found != SCHEMAS[kind]:
        raise VersionMismatch("%s schema_version %r; this build reads %d" % (kind, found, SCHEMAS[kind]))
    if "api_version" in record and api_major(record["api_version"]) != api_major():
        raise VersionMismatch("%s api_version %r; this build speaks %s" % (kind, record["api_version"], API_VERSION))
    return record

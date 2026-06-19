import os
import json
from collections import Counter, defaultdict

ROOT_DIR = "FT_Data"
MAX_EXAMPLES_PER_FILE = 5
MAX_GLOBAL_EXAMPLES = 30


def iter_jsonl_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".jsonl"):
                yield os.path.join(dirpath, name)


def type_name(x):
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, int):
        return "int"
    if isinstance(x, float):
        return "float"
    if isinstance(x, str):
        return "str"
    if isinstance(x, list):
        return "list"
    if isinstance(x, dict):
        return "dict"
    return type(x).__name__


def record_schema(obj):
    if not isinstance(obj, dict):
        return {"__root__": type_name(obj)}

    schema = {}
    for k, v in obj.items():
        if k == "messages" and isinstance(v, list):
            msg_schema = []
            for msg in v:
                if isinstance(msg, dict):
                    msg_schema.append({mk: type_name(mv) for mk, mv in msg.items()})
                else:
                    msg_schema.append(type_name(msg))
            schema[k] = msg_schema
        else:
            schema[k] = type_name(v)
    return schema


def validate_record(obj):
    errors = []

    if not isinstance(obj, dict):
        return [f"root is {type_name(obj)} instead of dict"]

    keys = set(obj.keys())
    if keys != {"messages"}:
        missing = {"messages"} - keys
        extra = keys - {"messages"}
        if missing:
            errors.append(f"missing keys: {sorted(missing)}")
        if extra:
            errors.append(f"unexpected keys: {sorted(extra)}")

    if "messages" not in obj:
        return errors or ["missing messages key"]

    messages = obj["messages"]
    if not isinstance(messages, list):
        errors.append(f"messages is {type_name(messages)} instead of list")
        return errors

    if len(messages) != 2:
        errors.append(f"messages length is {len(messages)} instead of 2")

    expected_roles = ["user", "assistant"]
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"messages[{i}] is {type_name(msg)} instead of dict")
            continue

        msg_keys = set(msg.keys())
        if msg_keys != {"role", "content"}:
            missing = {"role", "content"} - msg_keys
            extra = msg_keys - {"role", "content"}
            if missing:
                errors.append(f"messages[{i}] missing keys: {sorted(missing)}")
            if extra:
                errors.append(f"messages[{i}] unexpected keys: {sorted(extra)}")

        if "role" in msg:
            if not isinstance(msg["role"], str):
                errors.append(f"messages[{i}].role is {type_name(msg['role'])} instead of str")
            elif i < 2 and msg["role"] != expected_roles[i]:
                errors.append(f"messages[{i}].role is {msg['role']!r}, expected {expected_roles[i]!r}")

        if "content" in msg and not isinstance(msg["content"], str):
            errors.append(f"messages[{i}].content is {type_name(msg['content'])} instead of str")

    return errors


def main():
    jsonl_files = list(iter_jsonl_files(ROOT_DIR))

    if not jsonl_files:
        print(f"No .jsonl files found under: {ROOT_DIR}")
        return

    global_reference_schema = None
    schema_by_file = {}
    issues = []
    file_stats = {}

    for file_path in jsonl_files:
        line_count = 0
        valid_count = 0
        file_schema = None
        first_valid_seen = False

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                raw = line.strip()

                if not raw:
                    issues.append((file_path, line_num, "empty line"))
                    continue

                try:
                    obj = json.loads(raw)
                except Exception as e:
                    issues.append((file_path, line_num, f"invalid JSON: {e}"))
                    continue

                if not first_valid_seen:
                    file_schema = record_schema(obj)
                    first_valid_seen = True
                    if global_reference_schema is None:
                        global_reference_schema = file_schema

                valid_count += 1

                errs = validate_record(obj)
                if errs:
                    for err in errs:
                        issues.append((file_path, line_num, err))

        schema_by_file[file_path] = file_schema
        file_stats[file_path] = {"lines": line_count, "valid": valid_count}

        if file_schema is None:
            issues.append((file_path, 0, "no valid JSON object found in file"))

    print("=" * 100)
    print("FILES SCANNED:", len(jsonl_files))
    print("REFERENCE FILE:", jsonl_files[0])
    print("REFERENCE SCHEMA:", global_reference_schema)
    print("=" * 100)

    if not issues:
        print("All JSONL files share the same deep structural schema.")
        return

    print(f"Found {len(issues)} issue(s).")
    print()

    shown = 0
    per_file_counter = Counter()
    for file_path, line_num, msg in issues:
        per_file_counter[file_path] += 1
        if shown < MAX_GLOBAL_EXAMPLES:
            location = f"{file_path}"
            if line_num > 0:
                location += f":{line_num}"
            print(f"- {location} -> {msg}")
            shown += 1

    if len(issues) > MAX_GLOBAL_EXAMPLES:
        print(f"... {len(issues) - MAX_GLOBAL_EXAMPLES} more issue(s) not shown")

    print()
    print("Issue count by file:")
    for fp, n in per_file_counter.most_common():
        print(f"- {fp}: {n}")

    print()
    print("File-level summary:")
    for fp in jsonl_files:
        stats = file_stats.get(fp, {})
        schema = schema_by_file.get(fp)
        print(f"- {fp}")
        print(f"  lines={stats.get('lines', 0)}, valid_json_objects={stats.get('valid', 0)}")
        print(f"  first_valid_schema={schema}")

if __name__ == "__main__":
    main()
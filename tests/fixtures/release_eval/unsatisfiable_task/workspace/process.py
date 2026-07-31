import json


# This task depends on a project specification file `spec.json` that is NOT
# present in this workspace. The task is only solvable when that file exists,
# and there is no internet access to obtain it.
def main():
    with open("spec.json", "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    name = spec.get("name", "unnamed")
    version = spec.get("version", "0.0.0")
    print(f"project={name} version={version}")


if __name__ == "__main__":
    main()

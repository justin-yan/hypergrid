NAME:='hypergrid'
DEV_IMAGE:='ghcr.io/iomorphic/image/dev-py:latest'
SRC_FOLDER:='src'
TEST_FOLDER:='tests'



@default:
    just --list

@init:
    uv lock --check-exists && echo "Lockfile already exists" || just lock
    just sync

lock UPGRADE="noupgrade" PACKAGE="":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{UPGRADE}}" = "--upgrade" ] && [ -n "{{PACKAGE}}" ]; then
        uv lock --upgrade-package "{{PACKAGE}}"
    elif [ "{{UPGRADE}}" = "--upgrade" ] || [ "{{UPGRADE}}" = "-U" ]; then
        uv lock --upgrade
    else
        uv lock
    fi

sync FORCE="noforce":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{FORCE}}" = "--force" ]  || [ "{{FORCE}}" = "-f" ]; then
        rm -rf {{justfile_directory()}}/.venv
    fi
    uv sync --all-extras --frozen

@build:
    uv build

@verify: lint typecheck test
    echo "Done with Verification"

@lint:
    uv run --no-sync ruff check {{SRC_FOLDER}} {{TEST_FOLDER}}
    uv run --no-sync ruff format --check {{SRC_FOLDER}} {{TEST_FOLDER}}

@typecheck:
    uv run --no-sync mypy --explicit-package-bases -p {{NAME}}
    uv run --no-sync mypy --allow-untyped-defs tests

# Run tests. Optionally specify a specific test target e.g. `just test tests/path/to/test.py::test_name`
@test TARGET=TEST_FOLDER:
    uv run --no-sync pytest --hypothesis-show-statistics {{TARGET}}

# Run tests with a specific mark e.g. `just testmark slow`
testmark MARK="" TARGET=TEST_FOLDER:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run --no-sync pytest --hypothesis-show-statistics -m '{{MARK}}' {{TARGET}}

@format:
    uv run --no-sync ruff check --fix-only {{SRC_FOLDER}} {{TEST_FOLDER}}
    uv run --no-sync ruff format {{SRC_FOLDER}} {{TEST_FOLDER}}

@stats:
    uv run --no-sync coverage run -m pytest {{TEST_FOLDER}}
    uv run --no-sync coverage report -m
    scc --by-file --include-ext py

@cicd-pr: init verify
    echo "PR is successful!"

@cicd-register:
    git diff --name-only HEAD^1 HEAD -G"^version" "pyproject.toml" | uniq | xargs -I {} sh -c 'just _register'

@_register: init build
    uv publish -u $PYPI_USERNAME -p $PYPI_PASSWORD dist/*

@repl:
    uv run --no-sync python

@shell:
    #!/usr/bin/env bash
    pipenv shell

@run +COMMAND:
    uv run --no-sync {{COMMAND}}

######
## Custom Section Begin
######

######
## Custom Section End
######

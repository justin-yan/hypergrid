NAME:='hypergrid'
DEV_IMAGE:='ghcr.io/iomorphic/image/dev-py:latest'
SRC_FOLDER:='src'
TEST_FOLDER:='tests'



@default:
    just --list

@init:
    [ -f uv.lock ] && echo "Lockfile already exists" || uv lock
    uv sync

@build:
    uv build

@verify: lint typecheck test
    echo "Done with Verification"

@lint:
    uv run ruff check {{SRC_FOLDER}} {{TEST_FOLDER}}
    uv run ruff format --check {{SRC_FOLDER}} {{TEST_FOLDER}}

@typecheck:
    uv run mypy --explicit-package-bases -p {{NAME}}
    uv run mypy --allow-untyped-defs tests

@test:
    uv run pytest --hypothesis-show-statistics {{TEST_FOLDER}}

@format:
    uv run ruff check --fix-only {{SRC_FOLDER}} {{TEST_FOLDER}}
    uv run ruff format {{SRC_FOLDER}} {{TEST_FOLDER}}

@stats:
    uv run coverage run -m pytest {{TEST_FOLDER}}
    uv run coverage report -m
    scc --by-file --include-ext py

# docker host-mapped venv cannot be shared for localdev; container modified files not remapped to host user
virt SUBCOMMAND FORCE="noforce":
    #!/usr/bin/env bash
    if [ "{{FORCE}}" = "--force" ]  || [ "{{FORCE}}" = "-f" ]; then
        docker container prune --force
        docker volume rm --force {{NAME}}_pyvenv
    fi
    docker run -i -v `pwd`:`pwd` -v {{NAME}}_pyvenv:`pwd`/.venv -w `pwd` {{DEV_IMAGE}} just init {{SUBCOMMAND}}

@cicd-pr: init verify
    echo "PR is successful!"

@cicd-register:
    git diff --name-only HEAD^1 HEAD -G"^version" "pyproject.toml" | uniq | xargs -I {} sh -c 'just _register'

@_register: init build
    uv publish -u $PYPI_USERNAME -p $PYPI_PASSWORD dist/*

@lock:
    uv lock

@sync:
    uv sync

@repl:
    uv run python

######
## Custom Section Begin
######

######
## Custom Section End
######

PACKAGE:='hypergrid'
DEV_IMAGE:='ghcr.io/iomorphic/dev-python:latest'
SRC_FOLDER:='src'
TEST_FOLDER:='tests'



@default:
    just --list

@verify: lint typecheck test
    echo "Done with Verification"

@pr: init verify
    echo "PR is successful!"

@build:
    pipenv run python -m build

@register:
    git diff --name-only HEAD^1 HEAD -G"version" "pyproject.toml" | cut -d "/" -f2 | uniq | xargs -I {} sh -c 'just _register'

@_register: init build
    pipenv run twine upload --repository-url $PY_PRIVATE_REPO_URL -u $PY_PRIVATE_REPO_USERNAME -p $PY_PRIVATE_REPO_PASSWORD dist/*

@init:
    [ -f Pipfile.lock ] && echo "Lockfile already exists" || pipenv lock
    pipenv sync --dev

@docker SUBCOMMAND:
    echo "TODO: figure out how to run this with your local VENV"
    docker run -i -v `pwd`:`pwd` -w `pwd` {{DEV_IMAGE}} just {{SUBCOMMAND}}

@lint:
    echo "TODO: replace with ruff"
    pipenv run flake8 {{SRC_FOLDER}}
    pipenv run isort . --check --diff

typecheck:
    pipenv run mypy --explicit-package-bases -p {{PACKAGE}}
    pipenv run mypy --allow-untyped-defs tests

test:
    pipenv run pytest --hypothesis-show-statistics {{TEST_FOLDER}}

format:
    pipenv run black --verbose {{SRC_FOLDER}} {{TEST_FOLDER}}
    pipenv run isort .


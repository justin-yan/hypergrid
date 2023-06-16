PACKAGE:='hypergrid'
DEV_IMAGE:='ghcr.io/iomorphic/dev-py:latest'
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
    git diff --name-only HEAD^1 HEAD -G"^version" "pyproject.toml" | uniq | xargs -I {} sh -c 'just _register'

@_register: init build
    pipenv run twine upload -u $PYPI_USERNAME -p $PYPI_PASSWORD dist/*

@init:
    [ -f Pipfile.lock ] && echo "Lockfile already exists" || pipenv lock
    pipenv sync --dev

@docker SUBCOMMAND:
    echo "TODO: figure out how to run this with your local VENV"
    docker run -i -v `pwd`:`pwd` -w `pwd` {{DEV_IMAGE}} just {{SUBCOMMAND}}

@lint:
    pipenv run ruff check {{SRC_FOLDER}} {{TEST_FOLDER}}

@typecheck:
    pipenv run mypy --explicit-package-bases -p {{PACKAGE}}
    pipenv run mypy --allow-untyped-defs tests

@test:
    pipenv run pytest --hypothesis-show-statistics {{TEST_FOLDER}}

@format:
    pipenv run ruff check --fix {{SRC_FOLDER}} {{TEST_FOLDER}}
    pipenv run black --verbose {{SRC_FOLDER}} {{TEST_FOLDER}}

@stats:
    pipenv run coverage run -m pytest {{TEST_FOLDER}}
    pipenv run coverage report -m
    scc --by-file --include-ext py

@crossverify:
    #!/usr/bin/env bash
    set -euxo pipefail

    for py in 3.8.15 3.9.15 3.10.8 3.11.3
    do
        pyenv install -s $py
        pyenv local $py
        python -m venv /tmp/$py-crossverify
        source /tmp/$py-crossverify/bin/activate > /dev/null 2> /dev/null
        python --version
        pip -q install ruff mypy pytest hypothesis
        pip -q install -e .
        ruff check {{SRC_FOLDER}} {{TEST_FOLDER}}
        mypy --explicit-package-bases -p {{PACKAGE}}
        mypy --allow-untyped-defs tests
        pytest --hypothesis-show-statistics {{TEST_FOLDER}}
        deactivate > /dev/null 2> /dev/null
        rm -rf /tmp/$py-crossverify
        pyenv local --unset
    done

######
## Custom Section
######

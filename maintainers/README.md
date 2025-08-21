# Development install

```shell
uv sync
```

## Add dependencies

Regular or development dependencies are added as following
```shell
uv add numpy
uv add pytest --dev
```

## Run tests

```shell
uv run pytest
```

## Formatting and checking 

```shell
ruff format
ruff check --fix
```

## Bump version number 

```shell
bump-my-version bump patch
```

## Upgrading the dependencies

```shell
uv sync -U
```

## List the versions of the installed dependencies

```shell
uv tree --depth 1
```



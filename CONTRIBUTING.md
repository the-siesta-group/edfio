# Contributing to edfio
If you want to implement a new feature, fix an existing bug, or help improve edfio in any other way (such as adding or improving documentation), please consider submitting a [pull request](https://github.com/the-siesta-group/edfio/pulls) on GitHub.
It might be a good idea to open an [issue](https://github.com/the-siesta-group/edfio/issues) beforehand and discuss your planned contributions.

Before you start working on your contribution, please make sure to follow the guidelines described below.


## GitHub workflow
### Setup
- Create a [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) of the [repository](https://github.com/the-siesta-group/edfio).
- Clone the fork to your machine:

      git clone https://github.com/<your-username>/edfio

- Make sure your [username](https://docs.github.com/en/get-started/getting-started-with-git/setting-your-username-in-git) and [email](https://docs.github.com/en/github/setting-up-and-managing-your-github-user-account/managing-email-preferences/setting-your-commit-email-address#setting-your-commit-email-address-in-git) are configured to match your GitHub account.
- Add the original repository (also called _upstream_) as a remote to your local clone:

      git remote add upstream git@github.com:the-siesta-group/edfio.git


### Add a feature or fix a bug
- Create and switch to a new branch (use a self-explanatory branch name).
- Make changes and commit them.
- Push the changes to your remote fork.
- Create a [pull request (PR)](https://github.com/the-siesta-group/edfio/pulls).
- Add an entry to `CHANGELOG.md` (section "Unreleased") where you link to the corresponding PR and (if you desire) your account.


## Development environment
We suggest to use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for managing project dependencies, but any other standard-compliant tool will work too.

Create a virtual environment with Python 3.9 and install all dependencies:

    uv sync --python=3.9 --all-extras

**When using `uv`, prepend the commands below with `uv run` to make sure they are executed in the virtual environment!**


## Tests
To run the tests, in the project or package root execute

    pytest

Make sure all lines are covered by tests with

    pytest --cov


## Documentation
For docstrings, adhere to [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html), with the following exceptions/specifications:
- The maximum line length is `88`.
- For parameters that may take multiple types, pipe characters are used instead of the word `or`, for example `param_name : int | float`.


## Code style
We use the following tools for code style and quality:
- [Ruff](https://docs.astral.sh/ruff/) for formatting and linting
- [mypy](https://mypy.readthedocs.io/en/stable/) for static type checking

The development environment includes these tools, so you can either run them individually or use pre-commit.


## Pre-commit
Coding and documentation style are checked via a CI job.
To make sure your contribution passes those checks, you can use [`pre-commit`](https://pre-commit.com/) locally.
Install the hooks configured in `.pre-commit-config.yml` by running

    pre-commit install

inside your local clone.
After that, the checks required by the CI job will be run on all staged files when you commit – and abort the commit if any issues are found (in which case you should fix the issues and commit again).


## Releases
To release a new version, follow the steps below:
- Pick the correct version number according to [Semantic Versioning](https://semver.org/).
- In `CHANGELOG.md`, between the "Unreleased" heading and the latest changes, insert a line showing the version and release date, e.g., `## [0.2.0] - 2023-11-22`.
- Commit this change as e.g., `Release 0.2.0`
- [Create a new release](https://github.com/the-siesta-group/edfio/releases/new) on GitHub.
- Create a new tag where the target version is prefixed with a `v`, e.g., `v0.2.0`.
- Use the tag as the release title.
- The [release action](https://github.com/the-siesta-group/edfio/blob/main/.github/workflows/release.yml) takes care of the rest.

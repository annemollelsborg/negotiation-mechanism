init:
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	PATH="$$HOME/.local/bin:$$PATH" uv sync
	PATH="$$HOME/.local/bin:$$PATH" uv pip install -e ciceroscm-surrogate/ciceroscm/
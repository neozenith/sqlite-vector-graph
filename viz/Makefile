# Muninn Visualization Tool — Build & Dev targets
# All commands run from viz/ directory (invoked via make -C viz)

BACKEND_PORT ?= 8200
FRONTEND_PORT ?= 5280
AGENTIC_BACKEND_PORT ?= 8201
AGENTIC_FRONTEND_PORT ?= 5281

.PHONY: help install dev dev-agentic test test-api test-frontend test-e2e videos clean

help:                                              ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: install-python install-frontend           ## Install all dependencies

install-python:                                    ## Install Python deps
	uv sync --directory .

install-frontend:                                  ## Install frontend deps
	npm --prefix frontend install

dev: install                                       ## Start dev servers (human ports)
	npx --prefix frontend concurrently --kill-others \
		--names "backend,frontend" --prefix-colors "blue,green" \
		"uv run --directory . python -m server --port $(BACKEND_PORT)" \
		"VITE_API_PORT=$(BACKEND_PORT) npm --prefix frontend run dev -- --port $(FRONTEND_PORT)"

dev-agentic: install                               ## Start dev servers (agentic ports)
	npx --prefix frontend concurrently --kill-others \
		--names "backend,frontend" --prefix-colors "blue,green" \
		"uv run --directory . python -m server --port $(AGENTIC_BACKEND_PORT)" \
		"VITE_API_PORT=$(AGENTIC_BACKEND_PORT) npm --prefix frontend run dev -- --port $(AGENTIC_FRONTEND_PORT)"

test: test-api test-frontend                       ## Run all tests

test-api: install-python                           ## Python API tests (>90% coverage)
	uv run --directory . pytest -v

test-frontend: install-frontend                    ## TypeScript tests (>90% coverage)
	npm --prefix frontend run test -- --coverage

test-e2e: install                                  ## Playwright E2E tests
	npm --prefix frontend run test:e2e

# ── Video conversion ──────────────────────────────────────────────────

E2E_DIR = frontend/test-results

%.mp4: %.webm                                      ## Convert .webm to .mp4 (5x speed, 10fps)
	ffmpeg -i $< -vf "setpts=0.2*PTS,fps=10" -s 1280x720 -c:v libx264 -crf 20 -an -y $@

%.gif: %.webm                                      ## Convert .webm to .gif (5x speed, 10fps)
	ffmpeg -i $< -vf "setpts=0.2*PTS,fps=10,scale=1280:-1:flags=lanczos,palettegen" -y $@.palette.png
	ffmpeg -i $< -i $@.palette.png -filter_complex "setpts=0.2*PTS,fps=10,scale=1280:-1:flags=lanczos[x];[x][1:v]paletteuse" -y $@
	rm -f $@.palette.png

videos:                                            ## Convert all E2E videos
	$(MAKE) $(patsubst %.webm,%.mp4,$(wildcard $(E2E_DIR)/**/*.webm))
	$(MAKE) $(patsubst %.webm,%.gif,$(wildcard $(E2E_DIR)/**/*.webm))

# ── Clean ─────────────────────────────────────────────────────────────

clean:                                             ## Remove build artifacts
	rm -rf frontend/dist frontend/node_modules/.cache
	rm -rf frontend/screenshots frontend/test-results
	rm -rf .pytest_cache __pycache__ server/__pycache__ tests/__pycache__
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

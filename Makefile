.PHONY: run stop logs ps lint clean shell-gateway shell-agent help

## ── Lifecycle ─────────────────────────────────────────────────────────────────

run:           ## Build images and start all services in the background
	docker compose up --build -d

stop:          ## Stop all running services (preserves volumes)
	docker compose down

logs:          ## Tail logs from all services
	docker compose logs -f

ps:            ## Show service status and health
	docker compose ps

## ── Development helpers ───────────────────────────────────────────────────────

shell-gateway: ## Open a bash shell in the gateway container
	docker compose exec gateway bash

shell-agent:   ## Open a bash shell in the agent container
	docker compose exec agent bash

lint:          ## Run ruff linter against gateway and agent source
	docker compose run --rm gateway sh -c "pip install ruff -q && ruff check ."
	docker compose run --rm agent   sh -c "pip install ruff -q && ruff check ."

## ── Cleanup ───────────────────────────────────────────────────────────────────

clean:         ## Tear down containers, networks, and named volumes (destroys data)
	docker compose down -v --remove-orphans

## ── Help ──────────────────────────────────────────────────────────────────────

help:          ## Print this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

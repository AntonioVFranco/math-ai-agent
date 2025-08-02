# Docker Development Environment

This project includes a fully containerized development environment using Docker and Docker Compose. This ensures consistent development across all team members and eliminates "it works on my machine" issues.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)

## Quick Start

1. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone <repository-url>
   cd math-ai-agent
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

3. **Build and start the development environment:**
   ```bash
   docker-compose up --build -d
   ```

4. **Verify the container is running:**
   ```bash
   docker-compose ps
   ```

## Development Workflow

### Running Commands Inside the Container

To execute commands inside the running container:

```bash
# General command execution
docker-compose exec app <command>

# Examples:
docker-compose exec app python -m pytest
docker-compose exec app python src/main.py
docker-compose exec app pip install <package>
```

### Interactive Shell Access

To get an interactive shell inside the container:

```bash
docker-compose exec app bash
```

### Running Tests

```bash
# Run all tests
docker-compose exec app pytest

# Run specific test file
docker-compose exec app pytest tests/test_specific.py

# Run tests with verbose output
docker-compose exec app pytest -v
```

### Live Development

The following directories are mounted as volumes, so changes made locally are immediately reflected in the container:

- `./src` → `/app/src` (source code)
- `./data` → `/app/data` (data and benchmarks)
- `./examples` → `/app/examples` (example files)
- `./tests` → `/app/tests` (test files)

This means you can edit files in your local IDE and run them immediately in the container without rebuilding.

### Accessing the Gradio Interface

When you run the Gradio interface, it will be accessible at:
- **Local:** http://localhost:7860
- **Container:** The interface runs on port 7860 inside the container and is mapped to the same port on your host machine.

## Container Management

### Start the containers (in background):
```bash
docker-compose up -d
```

### Stop the containers:
```bash
docker-compose down
```

### Rebuild containers (after changing requirements.txt or Dockerfile):
```bash
docker-compose up --build -d
```

### View container logs:
```bash
docker-compose logs app
```

### Remove containers and volumes:
```bash
docker-compose down --volumes
```

## Troubleshooting

### Container won't start
- Check if the `.env` file exists and contains valid values
- Ensure Docker is running on your system
- Try rebuilding: `docker-compose up --build`

### Permission issues (Linux/Mac)
If you encounter permission issues with mounted volumes:
```bash
# Fix ownership of project files
sudo chown -R $USER:$USER .
```

### Port conflicts
If port 7860 is already in use, you can change it in `docker-compose.yml`:
```yaml
ports:
  - "8860:7860"  # Use port 8860 on host instead
```

### Package installation
To install additional packages:
```bash
# Add to requirements.txt, then rebuild
docker-compose up --build

# Or install temporarily in running container
docker-compose exec app pip install <package>
```

## Production Considerations

This Docker setup is optimized for development. For production deployment:

1. Remove volume mounts in `docker-compose.yml`
2. Change the CMD in `Dockerfile` to run your application
3. Consider using multi-stage builds for smaller production images
4. Use environment-specific docker-compose files (`docker-compose.prod.yml`)
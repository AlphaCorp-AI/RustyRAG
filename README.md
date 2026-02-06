# AlphaRust

A production-ready Rust web API built with Actix-web, featuring LLM integration via OpenRouter.

## Features

- 🚀 **High-performance async web server** powered by Actix-web
- 🤖 **LLM integration** with OpenRouter API support
- 📝 **OpenAPI/Swagger UI** for interactive API documentation
- ✅ **Request validation** with type-safe schemas
- 🔍 **Structured logging** with tracing
- 🌐 **CORS enabled** for cross-origin requests
- 🏗️ **Modular architecture** with clean separation of concerns

## Tech Stack

- **Web Framework**: Actix-web 4
- **Runtime**: Tokio (async)
- **Serialization**: Serde with validator
- **HTTP Client**: Reqwest
- **Documentation**: utoipa (OpenAPI 3.0)
- **Logging**: tracing + tracing-subscriber
- **Error Handling**: anyhow + thiserror

## Project Structure

```
src/
├── main.rs           # Application entry point
├── config.rs         # Configuration management
├── routes.rs         # Route definitions
├── errors.rs         # Error types and handlers
├── handlers/         # Request handlers
│   ├── health.rs     # Health check endpoint
│   └── chat.rs       # Chat/LLM endpoints
├── schemas/          # Request/response schemas
│   ├── requests.rs
│   └── responses.rs
├── services/         # Business logic
│   └── llm.rs        # LLM client integration
├── middleware/       # Custom middleware
└── db/               # Database layer (ready for future use)
```

## Getting Started

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- OpenRouter API key (get one at [openrouter.ai](https://openrouter.ai/))

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd alpharust
```

2. Copy the example environment file:
```bash
cp .env.example .env
```

3. Edit `.env` and add your API keys:
```env
OPENROUTER_API_KEY=sk-or-your-key-here
RUST_LOG=info
HOST=127.0.0.1
PORT=8080
```

4. Build and run:
```bash
cargo build --release
cargo run
```

The server will start at `http://127.0.0.1:8080`

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://127.0.0.1:8080/swagger-ui/
- **OpenAPI JSON**: http://127.0.0.1:8080/api-docs/openapi.json

## API Endpoints

### Health Check
```
GET /api/v1/health
```

### Chat (LLM)
```
POST /api/v1/chat
```

For detailed request/response schemas, see the Swagger UI.

## Development

### Running in Development Mode
```bash
cargo run
```

### Running with Debug Logging
```bash
RUST_LOG=debug cargo run
```

### Building for Production
```bash
cargo build --release
./target/release/alpharust
```

### Running Tests
```bash
cargo test
```

## Configuration

All configuration is managed through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Required |
| `HOST` | Server host | `127.0.0.1` |
| `PORT` | Server port | `8080` |
| `RUST_LOG` | Log level (error/warn/info/debug/trace) | `info` |

## Future Enhancements

- [ ] PostgreSQL integration with SQLx
- [ ] Authentication & authorization
- [ ] Rate limiting
- [ ] Caching layer
- [ ] WebSocket support
- [ ] Comprehensive test coverage

## License

[Choose your license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

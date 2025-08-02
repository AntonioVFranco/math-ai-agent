# 🧮 Math AI Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://hub.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/)

> **A specialized AI agent that excels at mathematical problem solving through intelligent reasoning, symbolic computation, and multi-layer verification.**

Math AI Agent combines the natural language understanding of OpenAI's GPT-4o with the precision of symbolic mathematics libraries to solve complex mathematical problems across multiple domains. Unlike traditional calculators or basic AI assistants, this agent provides **step-by-step solutions**, **verification of results**, and **interactive visualizations**.

## ✨ Why Math AI Agent?

- **🎯 Accuracy First**: Multi-layer verification system ensures solutions are mathematically correct
- **🧠 Intelligent Understanding**: Parses natural language and LaTeX to understand mathematical intent
- **📊 Visual Learning**: Generates interactive plots, graphs, and mathematical visualizations
- **🔍 Complete Solutions**: Provides step-by-step explanations, not just final answers
- **🚀 Production Ready**: Dockerized, tested, and documented for immediate deployment

## 🌟 Key Features

### **🔢 Multi-Domain Mathematical Solver**
- **Linear Algebra**: High-performance solver using NumPy/SciPy for matrix operations, SVD, eigenvalue decompositions, and matrix factorizations
- **Calculus & Analysis**: Symbolic derivatives, integrals, limits, and series using SymPy's computer algebra system
- **Optimization**: Advanced gradient descent solver capable of finding function minima - essential for Machine Learning applications
- **Statistics & Probability**: Comprehensive statistical analysis including hypothesis testing, distributions, correlation, and data analysis
- **Symbolic Mathematics**: Equation solving, expression simplification, factoring, and algebraic manipulation

### **✅ Multi-Layer Verification System** (Our Key Differentiator)
Unlike other mathematical tools, Math AI Agent cross-validates solutions using multiple verification methods:
- **Numerical Verification**: Tests symbolic solutions with random values
- **Algebraic Verification**: Confirms results through symbolic manipulation
- **Dimensional Analysis**: Validates unit consistency and mathematical properties
- **Cross-Method Validation**: Compares results across different solution approaches
- **Confidence Scoring**: Provides reliability metrics for each solution

### **📊 Integrated Visualizations**
Transform abstract mathematics into intuitive visual understanding:
- **Function Plotting**: 2D and 3D interactive plots with customizable ranges
- **Matrix Visualizations**: Heatmaps, eigenvalue plots, and decomposition displays
- **Optimization Paths**: Visual tracking of gradient descent convergence
- **Statistical Charts**: Histograms, box plots, scatter plots, and distribution curves
- **Dynamic Interactions**: Zoom, rotate, and explore mathematical objects

### **🤖 Intelligent Problem Parser**
Advanced natural language processing that understands mathematical intent:
- **LaTeX Support**: Full support for mathematical notation and expressions
- **Domain Classification**: Automatically identifies problem type and routes to appropriate solvers
- **Context Understanding**: Interprets word problems and mathematical descriptions
- **Confidence Metrics**: Reports parsing confidence and identifies ambiguous inputs

### **🐳 Production-Ready Infrastructure**
- **Docker Environment**: Fully containerized with consistent development setup
- **Performance Testing**: Comprehensive load testing suite with resource monitoring
- **API Integration**: RESTful interface for programmatic access
- **Comprehensive Documentation**: User guides, API reference, and developer documentation
- **Quality Assurance**: Extensive test suite with automated CI/CD validation

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/math-ai-agent.git
cd math-ai-agent

# Start with Docker Compose
docker-compose up

# Access the web interface at http://localhost:7860
```

### Option 2: Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your OpenAI API key

# Launch the application
python src/interface/app.py
```

### First Problem

1. **Get an OpenAI API Key**: Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Enter Your Key**: Click "Settings" in the web interface and paste your API key
3. **Solve a Problem**: Try `Find the eigenvalues of [[3, 1], [0, 2]]`
4. **Explore Results**: View the step-by-step solution, verification status, and visualizations

## 💡 Example Problems

### Linear Algebra
```
Calculate the SVD of matrix [[1,2,3],[4,5,6],[7,8,9]]
Find eigenvalues and eigenvectors of [[2,1],[1,2]]
Solve the linear system: 2x + y = 5, x - y = 1
```

### Calculus & Analysis
```
Find the derivative of sin(x^2) * cos(x)
Evaluate the integral: ∫(x * e^x) dx from 0 to 1
Find critical points of f(x,y) = x³ - 3xy² + y³
```

### Optimization
```
Minimize f(x,y) = (x-3)² + (y-2)² using gradient descent
Find the maximum of f(x) = -x² + 4x + 1
Solve: minimize x² + y² subject to x + y = 1
```

### Statistics
```
Perform t-test comparing samples [1,2,3,4,5] and [3,4,5,6,7]
Calculate correlation between study hours and grades
Test normality of data [15, 18, 16, 17, 14, 19, 20, 13]
```

## 📈 What Makes It Special

| Feature | Math AI Agent | Traditional Calculators | Basic AI Assistants |
|---------|---------------|-------------------------|-------------------|
| **Step-by-Step Solutions** | ✅ Detailed explanations | ❌ Final answer only | 🔶 Sometimes |
| **Verification System** | ✅ Multi-layer validation | ❌ No verification | ❌ No verification |
| **Visual Learning** | ✅ Interactive plots | ❌ Text only | ❌ Limited visuals |
| **Natural Language** | ✅ Full understanding | ❌ Syntax required | 🔶 Basic support |
| **Mathematical Rigor** | ✅ Symbolic precision | 🔶 Numerical only | ❌ Often inaccurate |
| **Domain Coverage** | ✅ Multi-domain expert | 🔶 Limited scope | 🔶 General knowledge |

## 🛠️ Architecture

Math AI Agent uses a sophisticated pipeline architecture:

```
User Input → Problem Parser → OpenAI Reasoning → Solution Planning
     ↓
Multi-Domain Solvers (Linear Algebra, Calculus, Optimization, Statistics)
     ↓
Multi-Layer Verification → Visualization Engine → Final Answer Synthesis
```

### Core Components
- **Engine**: Orchestrates the complete solution pipeline
- **Parser**: Intelligently classifies and routes mathematical problems  
- **Solvers**: Domain-specific mathematical computation modules
- **Verifier**: Multi-method solution validation system
- **Visualizer**: Interactive mathematical visualization generator
- **Interface**: User-friendly Gradio web application

## 📚 Documentation

- **[Quick Start Guide](docs/quickstart.rst)** - Get up and running in minutes
- **[Linear Algebra Tutorial](docs/tutorials/linear_algebra.rst)** - Matrix operations, decompositions, eigenvalues
- **[Optimization Tutorial](docs/tutorials/optimization.rst)** - Gradient descent, critical points, optimization
- **[Statistics Tutorial](docs/tutorials/statistics.rst)** - Hypothesis testing, distributions, analysis
- **[Developer Guide](docs/developer_guide.rst)** - Contributing, setup, and architecture
- **[API Reference](docs/api_reference.rst)** - Complete API documentation

## 🧪 Testing & Quality

Math AI Agent includes comprehensive testing infrastructure:

- **Unit Tests**: Individual component validation
- **Integration Tests**: End-to-end workflow testing  
- **Performance Tests**: Load testing with 50+ concurrent users
- **Acceptance Tests**: Mathematical accuracy validation
- **Benchmark Suite**: Performance and accuracy benchmarking

```bash
# Run the test suite
pytest

# Run performance tests
./scripts/run_performance_test.sh

# Run benchmarks
python tests/benchmark_runner.py
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository** and clone your fork
2. **Set up development environment**: `docker-compose up`
3. **Read the developer guide**: [docs/developer_guide.rst](docs/developer_guide.rst)
4. **Make your changes** with tests and documentation
5. **Submit a pull request** with clear description

### Development Areas
- **New Mathematical Domains**: Add solvers for geometry, number theory, etc.
- **Enhanced Verification**: Improve solution validation methods
- **Better Visualizations**: Create new plot types and interactions
- **Performance Optimization**: Improve speed and resource usage
- **User Experience**: Enhance the interface and workflows

## 📊 Performance

Math AI Agent is designed for production use:

- **Response Time**: < 15 seconds average for medium complexity problems
- **Concurrency**: Supports 50+ concurrent users
- **Memory Usage**: < 1GB under normal load
- **Accuracy**: 95%+ verification rate across mathematical domains
- **Uptime**: Tested for continuous operation

## 🔒 Security & Privacy

- **API Key Security**: Keys are never stored or logged
- **Isolated Execution**: Mathematical computations run in secure containers
- **Data Privacy**: No problem data is retained after processing
- **Input Validation**: Comprehensive sanitization of all inputs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **OpenAI**: For GPT-4o language model capabilities
- **SymPy**: For symbolic mathematics computation
- **NumPy/SciPy**: For numerical computing foundations
- **Gradio**: For the interactive web interface framework
- **Docker**: For containerization and deployment

## 📞 Support

- **Documentation**: Start with our comprehensive [documentation](docs/)
- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/your-username/math-ai-agent/issues)
- **Discussions**: Join community discussions on [GitHub Discussions](https://github.com/your-username/math-ai-agent/discussions)

---

**Ready to solve complex mathematical problems with AI?** [Get started now!](#-quick-start)

*Math AI Agent v1.0.0 - Transforming mathematical problem solving through intelligent automation.*
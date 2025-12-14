# Dynmarker Documentation

Welcome to the comprehensive documentation for the **ode-biomarker-project** (dynmarker) - a framework for dynamic biomarker discovery using ordinary differential equations and machine learning.

## Documentation Structure

### 📚 Core Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| **[onboarding.md](onboarding.md)** | Complete project overview and setup guide | New users, researchers |
| **[quick-start.md](quick-start.md)** | 15-minute getting started guide | Beginners, quick setup |
| **[architecture.md](architecture.md)** | System architecture and design patterns | Developers, advanced users |
| **[module-reference.md](module-reference.md)** | Detailed module and API reference | Developers, contributors |
| **[research-workflows.md](research-workflows.md)** | Practical research examples and workflows | Researchers, analysts |

### 🚀 Quick Navigation

**Getting Started** → Start with **[quick-start.md](quick-start.md)** for immediate setup

**Comprehensive Guide** → Read **[onboarding.md](onboarding.md)** for full understanding

**Research Examples** → Explore **[research-workflows.md](research-workflows.md)** for practical applications

**Technical Details** → Consult **[architecture.md](architecture.md)** and **[module-reference.md](module-reference.md)** for development

## Documentation Philosophy

This documentation follows a **progressive disclosure** approach:

1. **Quick Start**: Immediate setup and basic usage
2. **Onboarding**: Comprehensive project understanding  
3. **Workflows**: Practical research applications
4. **Technical Reference**: Detailed implementation specifics
5. **Architecture**: System design and extension patterns

## Key Features Documentation

### Dynamic Biomarker Discovery
- ODE-based dynamic modeling integration
- Multi-omics data fusion (proteomic, genomic, drug response)
- Advanced feature selection algorithms (MRMR, ReliefF, etc.)
- ML pipeline framework with parallel processing

### Framework Capabilities
- **Modular Architecture**: Easy extension and customization
- **Parallel Processing**: Efficient multi-core computation
- **Data Integration**: Flexible multi-source data handling
- **Benchmarking**: Comprehensive performance evaluation
- **Visualization**: Result analysis and interpretation tools

## Recommended Reading Order

### For New Researchers
1. **[quick-start.md](quick-start.md)** - Get running in 15 minutes
2. **[onboarding.md](onboarding.md)** - Understand project scope
3. **[research-workflows.md](research-workflows.md)** - Apply to your research

### For Developers/Contributors
1. **[architecture.md](architecture.md)** - Understand system design
2. **[module-reference.md](module-reference.md)** - API details
3. **[onboarding.md](onboarding.md)** - Project context

### For Advanced Users
1. **[research-workflows.md](research-workflows.md)** - Advanced applications
2. **[module-reference.md](module-reference.md)** - Technical specifics
3. **[architecture.md](architecture.md)** - Extension patterns

## Project Structure Reference

```
ode-biomarker-project/
├── dynmarker/              # Core framework
│   ├── GeneralPipeline.py  # Flexible execution framework
│   ├── EvaluationPipeline.py # ML evaluation pipeline
│   ├── FeatureSelection.py # Feature selection algorithms
│   └── DataLoader.py       # Data management
├── docs/                   # This documentation
├── scripts/               # Pipeline execution scripts
├── notebooks/             # Research analysis notebooks
├── thesis-notebooks/      # Thesis research documentation
└── project-*/            # Specific research projects
```

## Support and Resources

### Additional Resources
- **Source Code**: Inline documentation and comments
- **Example Scripts**: Pipeline scripts in root directory (`SYPipelineScript.py`, etc.)
- **Research Notebooks**: Analysis examples in various folders
- **Pre-processing Guidelines**: `pre-processing-guideline.md` in root

### Getting Help
- Review relevant documentation section
- Check example scripts and notebooks
- Examine test files for usage patterns
- Consult source code comments for implementation details

## Contributing to Documentation

We welcome contributions to improve this documentation:

1. **Clarity Improvements**: Better explanations and examples
2. **New Workflows**: Additional research application examples
3. **Technical Updates**: Reflect new features and changes
4. **Translation**: Multi-language support

Please follow the existing structure and style when contributing.

## Version Information

- **Documentation Version**: 1.0.0
- **Project Version**: 0.1.0
- **Last Updated**: December 2025
- **Compatible With**: Python >= 3.10

---

*This documentation aims to make dynmarker accessible to researchers at all levels, from beginners to advanced developers. Start with the quick start guide and progress based on your needs.*

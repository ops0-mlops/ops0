# Pull Request

## 📝 Description

<!-- Provide a clear and concise description of your changes -->

### What does this PR do?
<!-- 
- Fixes #(issue number)
- Adds new feature X
- Improves performance of Y
- Updates documentation for Z
-->

### Why is this change needed?
<!-- Explain the motivation behind this change -->

## 🔧 Type of Change

<!-- Mark the relevant option with an "x" -->

- [ ] 🐛 **Bug fix** (non-breaking change that fixes an issue)
- [ ] ✨ **New feature** (non-breaking change that adds functionality)
- [ ] 💥 **Breaking change** (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 **Documentation** (updates to documentation)
- [ ] 🎨 **Code style** (formatting, renaming, code organization)
- [ ] ♻️ **Refactoring** (code change that neither fixes a bug nor adds a feature)
- [ ] ⚡ **Performance** (code change that improves performance)
- [ ] ✅ **Tests** (adding missing tests or correcting existing tests)
- [ ] 🔧 **Tooling** (changes to build process, CI/CD, development tools)

## 🧪 How Has This Been Tested?

<!-- Describe the tests you ran to verify your changes -->

- [ ] **Unit tests** - `pytest tests/unit/`
- [ ] **Integration tests** - `pytest tests/integration/`
- [ ] **Manual testing** - Local pipeline execution
- [ ] **Example pipelines** - Tested with examples in `/examples/`

### Test Configuration:
- **Python version**: 
- **OS**: 
- **ops0 version**: 

### Test Commands Used:
```bash
# Example test commands you ran
pytest tests/unit/test_your_feature.py -v
ops0 run --local examples/your_example.py
```

## 📋 API Changes

<!-- If this PR introduces API changes, document them -->

### New APIs:
```python
# Example of new functions, decorators, or methods
@ops0.new_decorator
def example_function():
    pass
```

### Changed APIs:
```python
# Before
ops0.old_method(param1, param2)

# After  
ops0.new_method(param1, param2, new_param=default)
```

### Deprecated APIs:
<!-- List any APIs that are now deprecated -->

## 🔄 Migration Guide

<!-- If this is a breaking change, provide migration instructions -->

```python
# How users should update their code (if applicable)
# Before:
# old_code_example()

# After:
# new_code_example()
```

## 📸 Screenshots/Examples

<!-- Add screenshots or code examples that demonstrate the change -->

### Before:
```python
# Previous behavior/code
```

### After:
```python
# New behavior/code
```

## 📚 Documentation

<!-- Check all that apply -->

- [ ] **Code is self-documenting** with clear function/class names
- [ ] **Docstrings added/updated** for new or modified functions
- [ ] **Type hints added** for new functions and parameters
- [ ] **README updated** (if user-facing changes)
- [ ] **API documentation updated** (if API changes)
- [ ] **Examples added/updated** in `/examples/`
- [ ] **Changelog entry added** (for notable changes)

## 🔗 Related Issues

<!-- Link related issues -->

- Closes #(issue_number)
- Related to #(issue_number)  
- Part of #(issue_number)

## 🚨 Breaking Changes

<!-- If this introduces breaking changes, describe them -->

⚠️ **This PR introduces breaking changes**:

- [ ] **Function signature changes**:
- [ ] **Configuration changes**:
- [ ] **Behavior changes**:
- [ ] **Removed features**:

## ✅ Checklist

<!-- Mark completed items with "x" -->

### Code Quality
- [ ] **Self-review completed** - I have reviewed my own code
- [ ] **Code follows style guidelines** - Passes `black`, `isort`, `ruff`
- [ ] **Type hints added** - New code has appropriate type annotations
- [ ] **No debug code** - Removed print statements, debugger calls, etc.
- [ ] **Error handling** - Appropriate error handling for edge cases

### Testing
- [ ] **Tests added** - New functionality has corresponding tests
- [ ] **Tests pass** - All existing tests continue to pass
- [ ] **Coverage maintained** - Test coverage hasn't decreased significantly
- [ ] **Edge cases tested** - Considered and tested edge cases

### Documentation
- [ ] **Docstrings complete** - All public functions have docstrings
- [ ] **Comments added** - Complex logic is explained with comments
- [ ] **Examples work** - Any new examples actually run successfully
- [ ] **Documentation updated** - User-facing docs reflect changes

### ops0 Specific
- [ ] **Pipeline compatibility** - Works with existing ops0 pipelines
- [ ] **Decorator compatibility** - Compatible with existing `@ops0.step` usage
- [ ] **Storage compatibility** - Works with ops0 storage abstractions
- [ ] **Cloud deployment** - Tested or verified for cloud deployment
- [ ] **Performance impact** - Considered performance implications

### Git & Process
- [ ] **Commits are clean** - Logical, well-described commit messages
- [ ] **Branch is up to date** - Merged latest main/develop
- [ ] **No merge conflicts** - Clean merge possible
- [ ] **PR title follows convention** - Clear, descriptive title

## 🔮 Future Considerations

<!-- Optional: What should be considered for future iterations? -->

## 📞 Reviewer Notes

<!-- Optional: Specific areas you'd like reviewers to focus on -->

**Please pay special attention to**:
- [ ] Performance implications
- [ ] Security considerations
- [ ] API design decisions
- [ ] Error handling
- [ ] Documentation clarity

## 🙏 Additional Notes

<!-- Any additional information for reviewers -->

---

**For reviewers**: See our [Code Review Guidelines](https://docs.ops0.xyz/contributing/code-review) for review best practices.
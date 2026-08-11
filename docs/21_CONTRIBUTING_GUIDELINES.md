# Contributing Guidelines

To maintain the safety and integrity of the OMNIDRIVE autonomous stack, all developers must strictly adhere to the following guidelines. This codebase controls physical multi-ton vehicles; poor code can be lethal.

## 1. Branching Strategy (Git Flow)

We follow a strict Git branching model:
* `main`: The production-ready branch. Code here is actively deployed to physical hardware. DO NOT push directly to `main`.
* `develop`: The active integration branch. All feature branches merge here first for integration testing in CARLA.
* `feature/<name>`: For new features (e.g., `feature/radar-fusion`).
* `bugfix/<name>`: For resolving issues (e.g., `bugfix/memory-leak-rssm`).

## 2. Code Formatting and Linting

We enforce zero-tolerance linting rules using **Ruff** and **Black**.

Before you commit, run:
```bash
python -m ruff check --fix .
python -m black .
```
If the GitHub Actions CI pipeline detects any formatting errors, your Pull Request will be automatically rejected.

## 3. Writing Tests

Every new module **must** include a corresponding mathematical unit test in the `tests/` directory.

### Rules for Deep Learning Tests:
1. **Never use real weights in unit tests:** Tests must run in <1 second. Initialize tiny, dummy versions of networks (e.g., `depth=2`, `embed_dim=16`) instead of loading a 4GB checkpoint.
2. **Test Tensor Shapes:** Explicitly assert the input and output sizes of your network.
   ```python
   assert output.shape == (Batch, Tokens, Embed_Dim)
   ```
3. **Test Gradient Flow:** Ensure that `loss.backward()` actually populates gradients for your network parameters. If you detach a tensor incorrectly, the AI will silently fail to learn.

## 4. Submitting a Pull Request (PR)

1. Ensure your branch is updated with the latest `develop`.
2. Open a PR targeting `develop`.
3. In the PR description, explicitly state:
   * **What** the change does.
   * **Why** it was necessary.
   * **Mathematical impact:** (e.g., "Changes RL discount factor gamma from 0.99 to 0.995 to prioritize long-term route planning").
4. Wait for the `Integration Tests` to pass on the GitHub Actions runner (this takes ~10 minutes as it boots a virtual GPU).
5. At least one Senior AI Engineer must approve the PR before merging.

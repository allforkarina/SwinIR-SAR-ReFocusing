# Official reference material

The independent implementation in `swinir/` never imports this directory at
runtime. This directory is reserved for development-time structure and
numerical equivalence checks.

Pinned upstream:

- Repository: <https://github.com/JingyunLiang/SwinIR>
- Commit: `6545850fbf8df298df73d81f3e8cba638787c8bd`
- License: Apache-2.0

To enable the optional equivalence test, download these files at the pinned
commit:

```text
references/network_swinir.py
references/main_test_swinir.py
references/LICENSE
```

Raw source URLs:

```text
https://raw.githubusercontent.com/JingyunLiang/SwinIR/6545850fbf8df298df73d81f3e8cba638787c8bd/models/network_swinir.py
https://raw.githubusercontent.com/JingyunLiang/SwinIR/6545850fbf8df298df73d81f3e8cba638787c8bd/main_test_swinir.py
https://raw.githubusercontent.com/JingyunLiang/SwinIR/6545850fbf8df298df73d81f3e8cba638787c8bd/LICENSE
```

Then run:

```bash
python scripts/compare_official.py
pytest tests/test_official_equivalence.py
```

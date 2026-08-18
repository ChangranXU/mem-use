# 开发

## 测试

```bash
uv venv .venv && uv pip install pytest tiktoken tqdm   # 首次，在 checkout 内执行
.venv/bin/python -m pytest tests -q
```

测试不走网络——评审与提取客户端都打桩；夹具为
`tests/annotated_fixtures.py` 合成的 schema-v6 轨迹。

## 项目结构

- `util.py` — 共享工具（原子写产物、安全打印）
- `memgraph/` — 注解索引、解析、链接、HTML 渲染、判定合并、CLI
  （`python3 -m memgraph`）
- `injudge/` — 步骤流、注入区间、分级评审客户端、超预算扫描
  （`scan.py`）、gold 边车（`gold.py`）、聚合、报告、CLI
  （`python3 -m injudge`）
- `prompts/` — 所有 LLM 提示词文件，按工具/用途分目录（经
  `injudge/prompts.py` 加载；代码中绝不内联提示词）
- `tests/` — pytest 测试套件 + 共享的 schema-v6 合成轨迹构造器

## 贡献约定

工作约定——代码风格、commit message 格式、产物契约——见仓库根目录的
`AGENTS.md`。

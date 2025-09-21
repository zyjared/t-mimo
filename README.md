# t-mimo

> [!TIP]
> 进行中...

## 转换为 xlsx

`/data/` 目录结构：

```txt
data/
├── raw/
│   ├── 1.mat
│   └── 2.mat
├── processed/ # 通过 cli 转换
│   ├── 1.xlsx
│   └── 2.xlsx
└── ...
```

可以通过命令将数据转换为 `.xlsx`：

```bash
python -m src.preprocess.convert
```


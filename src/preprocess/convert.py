from pathlib import Path
from typing import Callable
from scipy.io import loadmat
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

from ..logger import logger
from ..constants import RAW_DIR, PROCESSED_DIR

from .loader import load_raw_data, ensure_suffix


def convert_single_file_worker(args):
    """
    用于多进程的单个文件转换函数
    """
    filename, raw_dir, processed_dir, suffix = args

    try:
        # 重新导入必要的模块（多进程需要）
        from scipy.io import loadmat
        import pandas as pd
        from pathlib import Path

        # 构建完整路径
        src_path = Path(raw_dir) / filename
        dst_path = Path(processed_dir) / ensure_suffix(
            filename.replace('.mat', ''),
            '.xlsx'
        )

        # 确保目标目录存在
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # 加载.mat文件
        raw_data = loadmat(str(src_path))

        # 提取数据变量
        data_vars = {
            k: v for k, v in raw_data.items() if not k.startswith('__')
        }

        if not data_vars:
            return f"ERROR: {filename} - 没有找到数据变量"

        # 创建Excel文件
        with pd.ExcelWriter(dst_path, engine='openpyxl') as writer:
            for var_name, var_data in data_vars.items():
                if var_data.ndim == 1:
                    df = pd.DataFrame({var_name: var_data})
                elif var_data.ndim == 2:
                    df = pd.DataFrame(
                        var_data,
                        columns=[
                            f"{var_name}_{i}" for i in range(var_data.shape[1])
                        ]
                    )
                else:
                    df = pd.DataFrame({var_name: var_data.flatten()})

                df.to_excel(writer, sheet_name=var_name, index=False)

        return f"SUCCESS: {filename} -> {dst_path}"

    except Exception as e:
        return f"ERROR: {filename} - {str(e)}"


def convert_raw_to_xlsx(
    src: str,
    *,
    suffix: str = ".mat",
    loader: Callable = loadmat
):
    """
    将原始数据转换为 xlsx,并保存到 processed_dir,保持文件名不变
    """
    dst = PROCESSED_DIR.joinpath(
        ensure_suffix(
            src.replace(".mat", ""),
            ".xlsx"
        )
    )

    raw = load_raw_data(src, suffix=suffix, loader=loader)
    if raw is None:
        return None

    # 从.mat文件中提取数据
    # raw 是一个字典，包含所有变量
    data_vars = {
        k: v for k, v in raw.items() if not k.startswith('__')
    }

    if not data_vars:
        logger.warning(f"文件 {src} 中没有找到数据变量")
        return None

    # 确保目标目录存在
    dst.parent.mkdir(parents=True, exist_ok=True)

    # 创建Excel写入器
    with pd.ExcelWriter(dst, engine='openpyxl') as writer:
        for var_name, var_data in data_vars.items():
            # 将numpy数组转换为DataFrame
            if var_data.ndim == 1:
                df = pd.DataFrame({var_name: var_data})
            elif var_data.ndim == 2:
                df = pd.DataFrame(
                    var_data,
                    columns=[
                        f"{var_name}_{i}" for i in range(var_data.shape[1])
                    ]
                )
            else:
                # 对于高维数据，展平后保存
                df = pd.DataFrame({var_name: var_data.flatten()})

            # 写入不同的sheet
            df.to_excel(writer, sheet_name=var_name, index=False)

    logger.debug(f"已转换 {src} -> {dst}")
    return dst


def convert_all_raw_to_xlsx_sync(
    loader: Callable = loadmat,
    *,
    suffix: str = ".mat",
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR
):
    """
    将所有原始数据转换为 xlsx,并保存到 processed_dir,保持文件名不变 (同步版本)
    """
    # 确保处理目录存在
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = list(raw_dir.glob(f"**/*{suffix}"))
    logger.info(f"找到 {len(files)} 个 {suffix} 文件")

    for file in files:
        filename = str(file.relative_to(raw_dir))
        logger.info(f"正在转换: {filename}")
        result = convert_raw_to_xlsx(filename, loader=loader)
        if result:
            logger.success(f"转换成功: {filename} -> {result}")
        else:
            logger.error(f"转换失败: {filename}")


async def convert_single_file_async(
    filename: str,
    *,
    suffix: str = ".mat",
    loader: Callable = loadmat,
    executor: ThreadPoolExecutor
):
    """
    异步转换单个文件
    """
    try:
        loop = asyncio.get_event_loop()

        # 在线程池中执行CPU密集型任务
        result = await loop.run_in_executor(
            executor,
            convert_raw_to_xlsx,
            filename,
            suffix,
            loader
        )

        if result:
            logger.success(f"转换成功: {filename} -> {result}")
        else:
            logger.error(f"转换失败: {filename}")

        return result
    except Exception as e:
        logger.error(f"转换异常: {filename} - {str(e)}")
        return None


def convert_all_raw_to_xlsx_multiprocess(
    *,
    suffix: str = ".mat",
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    max_workers: int = None
):
    """
    将所有原始数据转换为 xlsx,并保存到 processed_dir,保持文件名不变 (多进程版本)

    Args:
        suffix: 文件后缀
        raw_dir: 原始数据目录
        processed_dir: 处理后的数据目录
        max_workers: 最大并发进程数，默认为CPU核心数
    """
    # 确保处理目录存在
    processed_dir.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        max_workers = mp.cpu_count()

    files = list(raw_dir.glob(f"**/*{suffix}"))
    logger.info(f"找到 {len(files)} 个 {suffix} 文件")
    logger.info(f"使用 {max_workers} 个进程进行并行转换")

    # 准备任务参数
    tasks = []
    for file in files:
        filename = str(file.relative_to(raw_dir))
        tasks.append((filename, str(raw_dir), str(processed_dir), suffix))

    # 使用进程池执行器进行真正的并行处理，实时显示进度
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        logger.info("开始多进程转换...")

        # 提交所有任务
        future_to_task = {
            executor.submit(
                convert_single_file_worker,
                task
            ): task for task in tasks
        }

        success_count = 0
        error_count = 0
        completed = 0
        total = len(tasks)

        # 实时处理完成的任务
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                completed += 1

                if result.startswith("SUCCESS:"):
                    logger.success(f"[{completed}/{total}] {result}")
                    success_count += 1
                else:
                    logger.error(f"[{completed}/{total}] {result}")
                    error_count += 1

            except Exception as e:
                completed += 1
                task = future_to_task[future]
                logger.error(
                    f"[{completed}/{total}] ERROR: {task[0]} - {str(e)}"
                )
                error_count += 1

    logger.info(f"转换完成: 成功 {success_count} 个，失败 {error_count} 个")
    return success_count, error_count


def convert_all_raw_to_xlsx_async(
    loader: Callable = loadmat,
    *,
    suffix: str = ".mat",
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    max_workers: int = 4
):
    """
    将所有原始数据转换为 xlsx,并保存到 processed_dir,保持文件名不变 (线程池版本)

    Args:
        loader: 数据加载函数
        suffix: 文件后缀
        raw_dir: 原始数据目录
        processed_dir: 处理后的数据目录
        max_workers: 最大并发工作线程数
    """
    # 确保处理目录存在
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = list(raw_dir.glob(f"**/*{suffix}"))
    logger.info(f"找到 {len(files)} 个 {suffix} 文件")

    # 使用线程池执行器进行并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建任务
        tasks = []
        for file in files:
            filename = str(file.relative_to(raw_dir))
            logger.info(f"准备转换: {filename}")
            task = executor.submit(
                convert_raw_to_xlsx,
                filename,
                suffix=suffix,
                loader=loader
            )
            tasks.append((filename, task))

        # 等待所有任务完成
        logger.info(f"开始并发转换，使用 {max_workers} 个工作线程")
        success_count = 0
        error_count = 0

        for filename, task in tasks:
            try:
                result = task.result()
                if result:
                    logger.success(f"转换成功: {filename} -> {result}")
                    success_count += 1
                else:
                    logger.error(f"转换失败: {filename}")
                    error_count += 1
            except Exception as e:
                logger.error(f"转换异常: {filename} - {str(e)}")
                error_count += 1

        logger.info(f"转换完成: 成功 {success_count} 个，失败 {error_count} 个")
        return success_count, error_count


def convert_all_raw_to_xlsx(
    loader: Callable = loadmat,
    *,
    suffix: str = ".mat",
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    mode: str = "sync",
    max_workers: int = 4
):
    """
    将所有原始数据转换为 xlsx 的统一接口

    Args:
        loader: 数据加载函数
        suffix: 文件后缀
        raw_dir: 原始数据目录
        processed_dir: 处理后的数据目录
        mode: 处理模式 ("sync", "thread", "process")
        max_workers: 并发工作线程/进程数
    """
    if mode == "process":
        return convert_all_raw_to_xlsx_multiprocess(
            suffix=suffix,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            max_workers=max_workers
        )
    elif mode == "thread":
        return convert_all_raw_to_xlsx_async(
            loader=loader,
            suffix=suffix,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            max_workers=max_workers
        )
    else:  # sync
        return convert_all_raw_to_xlsx_sync(
            loader=loader,
            suffix=suffix,
            raw_dir=raw_dir,
            processed_dir=processed_dir
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="转换.mat文件为.xlsx"
    )

    parser.add_argument(
        "--mode",
        choices=["sync", "thread", "process"],
        default="process",
        help="处理模式: sync(同步), thread(线程池), process(多进程)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并发工作线程/进程数，默认为CPU核心数"
    )

    args = parser.parse_args()

    convert_all_raw_to_xlsx(mode=args.mode, max_workers=args.workers)

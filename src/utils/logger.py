from zyjared_color import red, green, yellow, magenta, cyan

class Logger:
    def __init__(self, name: str, *, debug: bool = False):
        self.name = name
        self._debug_enabled = debug
        self.colored_name = cyan(f'[{name}]')

    def _log(self, color_func, *values, sep=" ", end="\n", file=None, flush=False):
        """统一的日志输出方法"""
        if not values:
            return

        [first, *rest] = values

        print(
            f"{self.colored_name}",
            color_func(str(first)),
            *rest,
            end=end,
            file=file,
            flush=flush
        )

    def info(self, *values, sep=" ", end="\n", file=None, flush=False):
        self._log(cyan, *values, sep=sep, end=end, file=file, flush=flush)

    def success(self, *values, sep=" ", end="\n", file=None, flush=False):
        self._log(green, *values, sep=sep, end=end, file=file, flush=flush)

    def error(self, *values, sep=" ", end="\n", file=None, flush=False):
        self._log(red, *values, sep=sep, end=end, file=file, flush=flush)

    def warning(self, *values, sep=" ", end="\n", file=None, flush=False):
        self._log(yellow, *values, sep=sep, end=end, file=file, flush=flush)

    def debug(self, *values, sep=" ", end="\n", file=None, flush=False):
        if self._debug_enabled:
            self._log(magenta, *values, sep=sep, end=end, file=file, flush=flush)

    def child(self, name: str):
        return Logger(self.name + "." + name, debug=self._debug_enabled)

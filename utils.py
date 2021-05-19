import logging
from tqdm import tqdm
import datetime
import collections
import warnings
import sys
import torch.distributed as dist

FORMAT = "[%(asctime)s] [%(levelname)-8s] [%(node_rank)d ^ %(local_rank)d] [%(module)s] [%(filename)s:%(lineno)d] [%(message)s]"


class StructuredFormatter(logging.Formatter):
    converter = datetime.datetime.fromtimestamp
    tz = datetime.datetime.now().astimezone().tzinfo

    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        if isinstance(record.args, collections.abc.Mapping):
            for k in record.args:
                s += " [{}={}]".format(k, repr(record.args[k]))
        return s

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created, tz=self.tz)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s.%03d" % (t, record.msecs)
            s = "%s%s" % (s, ct.strftime("%z"))
        return s


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stdout)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class RankFilter(logging.Filter):
    def __init__(self, node_rank, local_rank) -> None:
        super().__init__()
        self.node_rank = node_rank
        self.local_rank = local_rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.node_rank = self.node_rank
        record.local_rank = self.local_rank
        return True


# TODO: Add File Handler
formatter = StructuredFormatter(FORMAT)


def getLoggerWithRank(name, node_rank, local_rank):
    basic_logger = logging.getLogger(name)
    basic_logger.setLevel(
        logging.INFO if local_rank in [-1, 0] else logging.WARNING)
    handler = TqdmLoggingHandler()
    handler.setFormatter(formatter)
    filter = RankFilter(node_rank, local_rank)
    basic_logger.handlers = [handler]
    basic_logger.filters = [filter]
    basic_logger.propagate = False
    return basic_logger


def redirect_warnings_to_logger(logger):
    def logwarn(message, category, filename, lineno, file=None, line=None):
        logger.warning(message, dict(filename=filename, lineno=lineno))

    warnings.showwarning = logwarn

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0
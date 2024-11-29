import os
import sys
import logging
import shutil
from logging import handlers

class CenteredLevelFormatter(logging.Formatter):
    def format(self, record):

        record.filename = record.filename[:20]  # set filename limit
        record.funcName = record.funcName[:25]  # set funcName limit
        #record.msg = record.msg[:300]  # set msg limit

        formatted_message = super().format(record)
        parts = formatted_message.split("|")

        widths = [25, 10, 10, 30, 25]  # asctime / PID / levelname / filename + lineno / funcName
        centered_parts = [
            part.strip().center(width) for part, width in zip(parts[:-1], widths)
        ]
        centered_parts.append(parts[-1]) # msg

        return " | ".join(centered_parts)

class Logger(logging.Logger):
    def __init__(self): 
        super().__init__(self)
        
        if os.path.exists("inference.log"):
            os.remove("inference.log")

        filename = "inference.log"
        backup_limit = 1
        # log format
        log_formatter = CenteredLevelFormatter(
             '%(asctime)s | PID %(process)d | %(levelname)s | %(filename)s :%(lineno)d | %(funcName)s |  %(message)s')

        # console write handler
        self.console_handler = logging.StreamHandler(sys.stdout)
        # file write handler
        self.file_handler = handlers.TimedRotatingFileHandler(filename=filename,
                                                              interval=1,
                                                              encoding='utf-8',
                                                              when='midnight',
                                                              backupCount=1)
        # stream handler
        self.stream_handler = logging.StreamHandler(sys.stdout)
        self.stream_handler_2 = logging.StreamHandler(sys.stderr)

        self.console_handler.setFormatter(log_formatter)
        self.file_handler.setFormatter(log_formatter)
        self.stream_handler.setFormatter(log_formatter)
        self.stream_handler_2.setFormatter(log_formatter)


        self.addHandler(self.console_handler)
        self.addHandler(self.file_handler)
        self.addHandler(self.stream_handler)
        self.addHandler(self.stream_handler_2)


        # log level 지정
        # INFO 로 설정시, logger.INFO 에 해당하는 내용만 출력
        # 미지정시 모든 내용 기록
        # DEBUG < INFO < WARN < ERROR < FATAL
        self.setLevel(logging.DEBUG)


if __name__ == "__main__":
    logger = Logger()
    logger.info("test")

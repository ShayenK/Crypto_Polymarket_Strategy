import os
import csv
from typing import Callable, Dict, Any, Type
from dataclasses import fields

class Tracking:
    def __init__(self, filepath:str, dataclass:Type, symbols:Dict[str,Any]):
        self.filepath:str = filepath
        self.dataclass:Type = dataclass
        self.symbols:Dict[str,Any] = symbols
        self.field_names = [field.name for field in fields(dataclass)]
        self.__write_csv_headers()

    def __write_csv_headers(self) -> None:

        # Check if Header Rows Exist to Write Headers
        write_headers = False
        if not os.path.exists(self.filepath):
            write_headers = True
        elif os.path.getsize(self.filepath) == 0:
            write_headers = True
        if write_headers:
            with open(self.filepath, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.field_names)
            print(f"INFO: initialized {self.filepath} csv with headers -> {len(self.field_names)} columns")

        return

    def logging(self, function:Type) -> Callable:

        def wrapper(*args:Any, **kwargs:Any) -> Any:
            
            append = function(*args, **kwargs)

            return append

        return wrapper
import os
import csv
import io
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from scapy.all import PcapReader# Import the necessary method from Scapy

from backend.controllers.files import FileHandler, UnsupportedFileExtensionError
from backend.models.file import File, SUPPORTED_FILE_EXTENSIONS

files_router = r = APIRouter()

@r.get("")
def fetch_files() -> list[File]:
    """
    Get the current files.
    """
    return FileHandler.get_current_files()

@r.post("")
async def add_file(file: UploadFile, fileIndex: str = Form(), totalFiles: str = Form()):
    file_extension = file.filename.split('.')[-1]

    if file_extension not in SUPPORTED_FILE_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "error": "UnsupportedFileExtensionError",
                "message": f"File extension '{file_extension}' is not supported.",
            },
        )

  
    res = await FileHandler.upload_file(file, file.filename, fileIndex, totalFiles)
    return res

@r.delete("/{file_name}")
def remove_file(file_name: str):
    """
    Remove a file.
    """
    try:
        FileHandler.remove_file(file_name)
    except FileNotFoundError:
        # Ignore the error if the file is not found
        pass
    return JSONResponse(
        status_code=200,
        content={"message": f"File {file_name} removed successfully."},
    )

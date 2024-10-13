import os
import csv
import io
from datetime import datetime
import pytz
from backend.models.file import SUPPORTED_FILE_EXTENSIONS, File, FileStatus
from backend.tasks.indexing import index_all
from scapy.all import PcapReader


class UnsupportedFileExtensionError(Exception):
    pass


class FileNotFoundError(Exception):
    pass


class FileHandler:
    @classmethod
    def get_current_files(cls):
        """
        Construct the list files by all the files in the data folder.
        """
        if not os.path.exists("data"):
            return []
        # Get all files in the data folder
        file_names = os.listdir("data")
        # Construct list[File]
        return [
            File(name=file_name, status=FileStatus.UPLOADED) for file_name in file_names
        ]
    
    @classmethod
    def get_local_time(cls):
        """
        Get the current time in the local timezone based on the server's configuration.
        """
        # Get the local timezone using the system's environment variable 'TZ', default to 'UTC' if not set
        local_tz = pytz.timezone(os.environ.get('TZ', 'UTC'))  # Use 'UTC' as default timezone if not set
        local_time = datetime.now(local_tz)
        return local_time

    @classmethod
    async def upload_file(
        cls, file, file_name: str, fileIndex: str, totalFiles: str
    ) -> File | UnsupportedFileExtensionError:
        """
        Upload a file to the data folder.
        """
        file_extension = file.filename.split('.')[-1]

        if file_extension not in SUPPORTED_FILE_EXTENSIONS:
            return UnsupportedFileExtensionError(
                f"File {file_name} with extension {file_name.split('.')[-1]} is not supported."
            )
        
        if not os.path.exists("data"):
            os.makedirs("data")

        if file_extension == "pcapng":
            contents = await file.read()
            pcap_data = PcapReader(io.BytesIO(contents))
            
            csv_output = io.BytesIO()
            csv_writer = csv.writer(io.TextIOWrapper(csv_output, newline='', encoding='utf-8'))

            # Write CSV headers
            csv_writer.writerow(["Timestamp", "Source", "Destination", "Protocol", "Length"])

            for packet in pcap_data:
                 if hasattr(packet, 'time') and hasattr(packet, 'src') and hasattr(packet, 'dst'):
                    # Convert packet time (EDecimal) to local timezone
                    packet_time = datetime.fromtimestamp(float(packet.time), tz=pytz.utc).astimezone(pytz.timezone(os.environ.get('TZ', 'UTC')))
                    proto = getattr(packet, 'proto', 'N/A')
                    length = len(packet)
                    csv_writer.writerow([packet_time.isoformat(), packet.src, packet.dst, proto, length])

            csv_output.seek(0)
            # Create the CSV filename based on the original file name
            csv_file_name = f"{os.path.splitext(file_name)[0]}.csv"
            
            with open(f"data/{csv_file_name}", "wb") as f:
                f.write(csv_output.getvalue())

            # Index the data only when it is the last file to upload
            if fileIndex == totalFiles:
                index_all()
            return File(name=csv_file_name, status=FileStatus.UPLOADED)

        # Handle normal file uploads
        with open(f"data/{file_name}", "wb") as f:
            f.write(await file.read())

        # Index the data
        if fileIndex == totalFiles:
            index_all()
        return File(name=file_name, status=FileStatus.UPLOADED)

    @classmethod
    def remove_file(cls, file_name: str) -> None:
        """
        Remove a file from the data folder.
        """
        os.remove(f"data/{file_name}")
        # Re-index the data
        index_all()
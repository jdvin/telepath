import requests

# URL of the file you want to download
file_url = "https://osf.io/b69u8/download"


# Sending a GET request to the file URL with stream=True
response = requests.get(file_url, stream=True)

# Check if the request was successful
if response.status_code == 200:
    # Get the total file size from the response headers
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 8192
    downloaded_size = 0

    # Specify the file name and path where you want to save the downloaded file
    with open("downloaded_file.extension", "wb") as file:
        # Download the file in chunks
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                file.write(chunk)
                downloaded_size += len(chunk)
                progress = (downloaded_size / total_size) * 100
                print(f"Download progress: {progress:.2f}%", end="\r")
    print("\nFile downloaded successfully")
else:
    print(f"Failed to download file. Status code: {response.status_code}")

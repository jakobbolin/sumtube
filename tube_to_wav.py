from yt_dlp import YoutubeDL

with open("URLS.txt", "r") as f:
    URLS = f.readlines()

path = "audio/"

ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': path + '%(title)s.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
}

with YoutubeDL(ydl_opts) as ydl:
    ydl.download(URLS)


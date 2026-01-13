
import os
import socket
import threading
import time
import uuid
import ipaddress
import qrcode
from flask import Flask, send_from_directory, render_template_string
from werkzeug.serving import make_server
from PIL import Image

def get_local_ip():
    """
    Returns a list of all non-loopback IP addresses found on network interfaces.
    """
    ips = set()
    
    # 1. Try Google DNS connection method (usually gets the primary outbound IP)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        if not ip.startswith('127.'):
            ips.add(ip)
    except Exception:
        pass

    # 2. Get all available interfaces
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None)
        for info in infos:
            ip = info[4][0]
            # IPv4 only for simplicity and wide compatibility
            if ':' not in ip and not ip.startswith('127.'):
                ips.add(ip)
    except Exception:
        pass
    
    if not ips:
        return ['127.0.0.1']
    
    # Sort to put common LAN prefixes first (192.168, 172., 10.) if multiple
    def sort_key(ip):
        if ip.startswith('192.168.'): return 0
        if ip.startswith('172.'): return 1
        if ip.startswith('10.'): return 2
        return 3
        
    return sorted(list(ips), key=sort_key)

def generate_qr_code(data):
    """
    Generates a QR code PIL Image from the given string data.
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    if hasattr(img, 'get_image'):
        return img.get_image()
    return img

class SharingServer:
    def __init__(self, file_paths):
        self.app = Flask(__name__)
        self.file_paths = file_paths
        self.file_map = {os.path.basename(p): p for p in file_paths}
        self.server = None
        self.thread = None
        self.port = 0 # 0 means random free port
        self.host = '0.0.0.0'
        self.token = uuid.uuid4().hex

        # Routes
        self.app.add_url_rule(f'/{self.token}/', 'index', self.index)
        self.app.add_url_rule(f'/{self.token}/download/<filename>', 'download', self.download)

    def index(self):
        items_html = ""
        for name in self.file_map.keys():
            items_html += f'''
                <div class="card">
                    <a href="download/{name}" target="_blank">
                        <img src="download/{name}" alt="{name}" loading="lazy">
                    </a>
                    <div class="card-body">
                        <div class="filename">{name}</div>
                        <a href="download/{name}" class="btn" download>Download</a>
                    </div>
                </div>
            '''

        return render_template_string(f'''
            <!doctype html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>Shared Photos</title>
                <style>
                    :root {{
                        --primary: #2563eb;
                        --primary-hover: #1d4ed8;
                        --bg: #f3f4f6;
                        --card-bg: #ffffff;
                        --text: #1f2937;
                        --text-light: #6b7280;
                    }}
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                        background-color: var(--bg);
                        color: var(--text);
                        margin: 0;
                        padding: 1.5rem;
                        line-height: 1.5;
                    }}
                    .container {{
                        max_width: 1000px;
                        margin: 0 auto;
                    }}
                    header {{
                        text-align: center;
                        margin-bottom: 2rem;
                    }}
                    h1 {{
                        font-size: 1.75rem;
                        font-weight: 800;
                        margin: 0 0 0.5rem 0;
                        color: #111827;
                    }}
                    p.subtitle {{
                        color: var(--text-light);
                        margin: 0;
                    }}
                    .gallery {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                        gap: 1.5rem;
                    }}
                    .card {{
                        background: var(--card-bg);
                        border-radius: 16px;
                        overflow: hidden;
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                        transition: transform 0.2s ease, box-shadow 0.2s ease;
                        display: flex;
                        flex-direction: column;
                    }}
                    .card:hover {{
                        transform: translateY(-4px);
                        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                    }}
                    .card a.img-link {{
                        display: block;
                        overflow: hidden;
                        height: 220px;
                    }}
                    .card img {{
                        width: 100%;
                        height: 100%;
                        object-fit: cover;
                        background-color: #e5e7eb;
                        transition: transform 0.3s ease;
                    }}
                    .card:hover img {{
                        transform: scale(1.05);
                    }}
                    .card-body {{
                        padding: 1.25rem;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        flex-grow: 1;
                    }}
                    .filename {{
                        font-size: 0.875rem;
                        color: var(--text-light);
                        margin-bottom: 1rem;
                        word-break: break-all;
                        text-align: center;
                    }}
                    .btn {{
                        background-color: var(--primary);
                        color: white;
                        font-weight: 600;
                        padding: 0.75rem 1.5rem;
                        border-radius: 9999px;
                        text-decoration: none;
                        transition: background-color 0.2s;
                        width: 100%;
                        text-align: center;
                        box-sizing: border-box;
                        display: block;
                    }}
                    .btn:hover {{
                        background-color: var(--primary-hover);
                    }}
                    .footer {{
                        margin-top: 3rem;
                        text-align: center;
                        font-size: 0.75rem;
                        color: var(--text-light);
                    }}
                    @media (max-width: 480px) {{
                        .gallery {{
                            grid-template-columns: 1fr;
                        }}
                        body {{
                            padding: 1rem;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <header>
                        <h1>Shared Photos</h1>
                        <p class="subtitle">Tap to view or download</p>
                    </header>
                    
                    <div class="gallery">
                        {items_html}
                    </div>

                    <div class="footer">
                        <p>Powered by Auto-Shutter</p>
                    </div>
                </div>
            </body>
            </html>
        ''')

    def download(self, filename):
        if filename in self.file_map:
            directory = os.path.dirname(self.file_map[filename])
            return send_from_directory(directory, filename, as_attachment=True)
        return "File not found", 404

    def start(self):
        # Find a free port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('0.0.0.0', 0))
        self.port = sock.getsockname()[1]
        sock.close()

        self.server = make_server(self.host, self.port, self.app)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        
        local_ips = get_local_ip() # Returns a list now
        urls = [f'http://{ip}:{self.port}/{self.token}/' for ip in local_ips]
        return urls

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.thread.join()

def serve_and_generate_qr(file_paths):
    """
    Starts a background server for the given files and returns:
    (qr_image_object, url_list, stop_function)
    """
    server = SharingServer(file_paths)
    urls = server.start()
    
    # Generate QR for the first (primary) IP
    qr_img = generate_qr_code(urls[0])
    
    return qr_img, urls, server.stop

if __name__ == '__main__':
    # Simple test
    import sys
    if len(sys.argv) > 1:
        files = sys.argv[1:]
        img, urls, stop = serve_and_generate_qr(files)
        print(f"Serving at: {urls[0]}")
        if len(urls) > 1:
            print(f"Alternative URLs: {', '.join(urls[1:])}")
            
        print("Press Ctrl+C to stop")
        img.show() # Display QR
        print("Press Ctrl+C to stop")
        img.show() # Display QR
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            stop()
            print("Server stopped")
    else:
        print("Usage: python simple_server_qr.py <file1> <file2> ...")

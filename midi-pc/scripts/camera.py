import io
import logging
import socket
import socketserver
from http import server
from gpiozero import LED
from threading import Condition
import signal
import sys
import time

from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

# Initialize LEDs
led_array = LED(17)    # Main LED (on pin 17)
error_led = LED(26)    # Red LED for errors (on pin 26)

# HTML page for the MJPEG streaming demo
PAGE = """\
<html>
<head>
<title>GRADIAN - Leia's sensor</title>
</head>
<body>
<h1>GRADIAN - Leia's sensor</h1>
<img src="stream.mjpg" width="640" height="640" />
</body>
</html>
"""

# Class to handle streaming output
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

# Class to handle HTTP requests
class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # Redirect root path to index.html
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            # Serve the HTML page
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            # Set up MJPEG streaming
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            # Handle 404 Not Found
            self.send_error(404)
            self.end_headers()

# Class to handle streaming server
class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

# Function to handle CTRL+C
def signal_handler(sig, frame):
    print("CTRL+C detected, stopping transmission...")
    stop_all()
    sys.exit(0)

# Register the signal handler for SIGINT (CTRL+C)
signal.signal(signal.SIGINT, signal_handler)

# Function to blink the red LED on errors
def blink_error_led(times=3, interval=0.5):
    for _ in range(times):
        error_led.on()
        time.sleep(interval)
        error_led.off()
        time.sleep(interval)

# Function to stop the camera and turn off the LEDs
def stop_all():
    try:
        picam2.stop_recording()
        blink_error_led(2)
    except Exception as e:
        logging.error(f"An error occurred while stopping the recording: {e}")
    led_array.off()
    error_led.off()

# Function to get the local IP address
def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to an external address (won't actually send any data)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
    except Exception as e:
        ip_address = '127.0.0.1'  # Fallback to localhost if unable to get IP
    return ip_address

# Create Picamera2 instance and configure it
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 640)}))
output = StreamingOutput()

try:
    picam2.start_recording(JpegEncoder(), FileOutput(output))

    # Get IP address and print the server URL
    ip_address = get_ip_address()
    print(f"Camera streaming available at: http://{ip_address}:8000")

    # Set up and start the streaming server
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    led_array.on()
    print("Server started. Press CTRL+C to stop.")
    server.serve_forever()

except Exception as e:
    logging.error(f"An error occurred: {e}")
    # Blink red LED on error
    blink_error_led()
    stop_all()

finally:
    # Ensure the camera and LEDs are properly stopped
    stop_all()

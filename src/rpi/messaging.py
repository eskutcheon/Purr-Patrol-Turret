"""
- Handles communication with the desktop server running with host/server.py
- Handles persistent network connections using context managers for reliability

send_frame_to_server(frame): Sends a frame to the server as a byte stream
receive_and_process_response(response, turret): Processes server responses to control the turret

server_connection(url): Manages a persistent connection to the desktop server
"""
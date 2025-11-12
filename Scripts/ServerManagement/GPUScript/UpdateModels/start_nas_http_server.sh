#!/bin/bash
# Start HTTP server on NAS for model distribution

NAS_HOST="root@dt-thpc-nas01"
NAS_PATH="/zfs/data/official-models-ckpt-tensordata"
NAS_IP="100.104.93.82"
HTTP_PORT="8000"

echo "Starting HTTP server on NAS..."
echo "  Host: $NAS_HOST"
echo "  Path: $NAS_PATH"
echo "  Bind IP: $NAS_IP"
echo "  Port: $HTTP_PORT"
echo ""

# Check if already running
echo "Checking if server is already running..."
EXISTING_PID=$(ssh -T "$NAS_HOST" "lsof -ti:$HTTP_PORT" 2>/dev/null)

if [ -n "$EXISTING_PID" ]; then
    echo "✅ HTTP server already running (PID: $EXISTING_PID)"
    echo "   URL: http://$NAS_IP:$HTTP_PORT"
    exit 0
fi

# Start the HTTP server
echo "Starting HTTP server..."
# Use setsid to completely detach the process from the terminal session
# -T disables pseudo-TTY allocation, -n redirects stdin from /dev/null
ssh -T -n "$NAS_HOST" "setsid sh -c 'cd $NAS_PATH && python3 -m http.server $HTTP_PORT --bind $NAS_IP </dev/null >/dev/null 2>&1 &'"

echo "   Server starting..."

# Wait for server to start
sleep 2

# Verify it's listening
echo "Verifying server is listening..."
VERIFY_PID=$(ssh -T "$NAS_HOST" "lsof -ti:$HTTP_PORT" 2>/dev/null)

if [ -z "$VERIFY_PID" ]; then
    echo "❌ Server started but not listening on port $HTTP_PORT"
    exit 1
fi

echo "✅ HTTP server started successfully"
echo "   URL: http://$NAS_IP:$HTTP_PORT"
echo "   PID: $VERIFY_PID"
echo ""
echo "To stop: ./stop_nas_http_server.sh"

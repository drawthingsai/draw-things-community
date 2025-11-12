#!/bin/bash
# Stop HTTP server on NAS for model distribution

NAS_HOST="root@dt-thpc-nas01"
HTTP_PORT="8000"

echo "Stopping HTTP server on NAS..."
echo "  Host: $NAS_HOST"
echo "  Port: $HTTP_PORT"
echo ""

# Find process listening on the port
echo "Checking for running server..."
EXISTING_PID=$(ssh "$NAS_HOST" "lsof -ti:$HTTP_PORT" 2>/dev/null)

if [ -z "$EXISTING_PID" ]; then
    echo "ℹ️  No HTTP server running on port $HTTP_PORT"
    exit 0
fi

echo "Found server process (PID: $EXISTING_PID)"

# Kill the process
echo "Stopping server..."
ssh "$NAS_HOST" "kill $EXISTING_PID" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "❌ Failed to stop server (PID: $EXISTING_PID)"
    echo "   Try: ssh $NAS_HOST 'kill -9 $EXISTING_PID'"
    exit 1
fi

# Wait a moment for graceful shutdown
sleep 1

# Verify it's stopped
VERIFY_PID=$(ssh "$NAS_HOST" "lsof -ti:$HTTP_PORT" 2>/dev/null)

if [ -n "$VERIFY_PID" ]; then
    echo "⚠️  Server still running, forcing shutdown..."
    ssh "$NAS_HOST" "kill -9 $VERIFY_PID" 2>/dev/null
    sleep 1

    # Final check
    FINAL_CHECK=$(ssh "$NAS_HOST" "lsof -ti:$HTTP_PORT" 2>/dev/null)
    if [ -n "$FINAL_CHECK" ]; then
        echo "❌ Failed to force stop server"
        exit 1
    fi
fi

echo "✅ HTTP server stopped successfully"
echo ""
echo "To restart: ./start_nas_http_server.sh"

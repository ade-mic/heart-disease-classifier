#!/bin/bash

# Start FastAPI Application
# Usage: ./start_fastapi.sh

echo "🚀 Starting Heart Disease Prediction FastAPI Server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Activate virtual environment
source venv/bin/activate

# Navigate to fastapi_app directory
cd fastapi_app

# Start the server
echo ""
echo "📍 Server will be available at:"
echo "   🌐 Main App:     http://localhost:8000"
echo "   📚 API Docs:     http://localhost:8000/docs"
echo "   📖 ReDoc:        http://localhost:8000/redoc"
echo "   ❤️  Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run with uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
